import socket, time

s = socket.socket()
s.connect(("127.0.0.1", 9002))

while True:
    s.send(b"0.3 0.0 0.0 0.0\n")
    time.sleep(0.05)
