#include <math.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <signal.h>

#include <webots/robot.h>
#include <webots/motor.h>
#include <webots/gps.h>
#include <webots/gyro.h>
#include <webots/inertial_unit.h>
#include <webots/camera.h>

#include "pid_controller.h"

#define TCP_PORT 9002
#define FLYING_ALTITUDE 1.0

static int server_fd = -1;
static int client_fd = -1;

float cmd_vx = 0, cmd_vy = 0, cmd_vz = 0, cmd_yaw = 0;

/* ================= TCP ================= */

static void init_tcp() {
  server_fd = socket(AF_INET, SOCK_STREAM, 0);

  struct sockaddr_in addr;
  addr.sin_family = AF_INET;
  addr.sin_port = htons(TCP_PORT);
  addr.sin_addr.s_addr = INADDR_ANY;

  bind(server_fd, (struct sockaddr *)&addr, sizeof(addr));
  listen(server_fd, 1);

  int flags = fcntl(server_fd, F_GETFL, 0);
  fcntl(server_fd, F_SETFL, flags | O_NONBLOCK);

  printf("[TCP] Listening on %d\n", TCP_PORT);
  signal(SIGPIPE, SIG_IGN);
}

static int frame_requested = 0;

static void handle_tcp() {
  if (client_fd < 0) {
    client_fd = accept(server_fd, NULL, NULL);
    if (client_fd >= 0) {
      int flags = fcntl(client_fd, F_GETFL, 0);
      fcntl(client_fd, F_SETFL, flags | O_NONBLOCK);
      frame_requested = 0;
      printf("[TCP] Client connected\n");
    }
    return;
  }

  char buf[256];
  int n = recv(client_fd, buf, sizeof(buf) - 1, MSG_DONTWAIT);
  if (n > 0) {
    buf[n] = '\0';
    /* Parse all lines in the buffer (may contain multiple messages) */
    char *line = strtok(buf, "\n");
    while (line != NULL) {
      if (strncmp(line, "FRAME", 5) == 0) {
        frame_requested = 1;
      } else {
        sscanf(line, "%f %f %f %f", &cmd_vx, &cmd_vy, &cmd_vz, &cmd_yaw);
      }
      line = strtok(NULL, "\n");
    }
  } else if (n == 0) {
    /* Client disconnected cleanly */
    close(client_fd);
    client_fd = -1;
    frame_requested = 0;
    printf("[TCP] Client disconnected\n");
  }
}

/* ================= MAIN ================= */

int main() {
  wb_robot_init();
  int timestep = wb_robot_get_basic_time_step();

  /* MOTORS */
  WbDeviceTag m1 = wb_robot_get_device("m1_motor");
  WbDeviceTag m2 = wb_robot_get_device("m2_motor");
  WbDeviceTag m3 = wb_robot_get_device("m3_motor");
  WbDeviceTag m4 = wb_robot_get_device("m4_motor");

  wb_motor_set_position(m1, INFINITY);
  wb_motor_set_position(m2, INFINITY);
  wb_motor_set_position(m3, INFINITY);
  wb_motor_set_position(m4, INFINITY);

  /* SENSORS */
  WbDeviceTag imu = wb_robot_get_device("inertial_unit");
  WbDeviceTag gps = wb_robot_get_device("gps");
  WbDeviceTag gyro = wb_robot_get_device("gyro");

  wb_inertial_unit_enable(imu, timestep);
  wb_gps_enable(gps, timestep);
  wb_gyro_enable(gyro, timestep);

  /* CAMERA */
  WbDeviceTag camera = wb_robot_get_device("vlm_camera");
  wb_camera_enable(camera, timestep);

  int cam_w = wb_camera_get_width(camera);
  int cam_h = wb_camera_get_height(camera);
  printf("[CAM] %dx%d\n", cam_w, cam_h);

  /* PID */
  actual_state_t actual_state = {0};
  desired_state_t desired_state = {0};

  gains_pid_t gains = {
    .kp_att_y = 1,
    .kd_att_y = 0.5,
    .kp_att_rp = 0.5,
    .kd_att_rp = 0.1,
    .kp_vel_xy = 2,
    .kd_vel_xy = 0.5,
    .kp_z = 10,
    .ki_z = 5,
    .kd_z = 5
  };

  init_pid_attitude_fixed_height_controller();
  motor_power_t motor_power;

  double height_desired = FLYING_ALTITUDE;
  double past_x = 0, past_y = 0, past_t = wb_robot_get_time();

  init_tcp();

  while (wb_robot_step(timestep) != -1) {
    double now = wb_robot_get_time();
    double dt = now - past_t;
    past_t = now;

    handle_tcp();

    const double *rpy = wb_inertial_unit_get_roll_pitch_yaw(imu);
    const double *g = wb_gps_get_values(gps);
    const double *gyro_v = wb_gyro_get_values(gyro);

    actual_state.roll = rpy[0];
    actual_state.pitch = rpy[1];
    actual_state.yaw_rate = gyro_v[2];
    actual_state.altitude = g[2];

    double vx = (g[0] - past_x) / dt;
    double vy = (g[1] - past_y) / dt;
    past_x = g[0];
    past_y = g[1];

    double cy = cos(rpy[2]);
    double sy = sin(rpy[2]);

    actual_state.vx =  vx * cy + vy * sy;
    actual_state.vy = -vx * sy + vy * cy;

    height_desired += cmd_vz * dt;

    desired_state.vx = cmd_vx;
    desired_state.vy = cmd_vy;
    desired_state.yaw_rate = cmd_yaw;
    desired_state.altitude = height_desired;

    pid_velocity_fixed_height_controller(
      actual_state, &desired_state, gains, dt, &motor_power
    );

    wb_motor_set_velocity(m1, -motor_power.m1);
    wb_motor_set_velocity(m2,  motor_power.m2);
    wb_motor_set_velocity(m3, -motor_power.m3);
    wb_motor_set_velocity(m4,  motor_power.m4);

    /* SEND CAMERA — Solo cuando el cliente lo pide (pull-based) */
    if (client_fd >= 0 && frame_requested) {
      frame_requested = 0;
      const unsigned char *img = wb_camera_get_image(camera);
      int header[2] = { cam_w, cam_h };

      ssize_t n1 = send(client_fd, header, sizeof(header), 0);
      if (n1 > 0) {
        /* Enviar imagen completa (blocking send) */
        size_t total = cam_w * cam_h * 4;
        size_t sent = 0;
        while (sent < total) {
          ssize_t n2 = send(client_fd, img + sent, total - sent, 0);
          if (n2 <= 0) {
            close(client_fd);
            client_fd = -1;
            printf("[TCP] Client disconnected (send error)\n");
            break;
          }
          sent += n2;
        }
      } else {
        close(client_fd);
        client_fd = -1;
        printf("[TCP] Client disconnected (header send error)\n");
      }
    }
  }

  wb_robot_cleanup();
  return 0;
}

