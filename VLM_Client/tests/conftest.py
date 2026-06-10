import sys
import unittest.mock as _mock

_langchain_openai = _mock.MagicMock()
_langchain_core_msgs = _mock.MagicMock()
sys.modules.setdefault("langchain_openai", _langchain_openai)
sys.modules.setdefault("langchain_core", _mock.MagicMock())
sys.modules.setdefault("langchain_core.messages", _langchain_core_msgs)
sys.modules.setdefault("langchain_core.tools", _mock.MagicMock())
sys.modules.setdefault("langchain", _mock.MagicMock())
