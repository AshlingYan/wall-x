import socket
import pickle
from threading import Thread, Event

from utils.data_handler import debug_print

class BiSocket:
    '''
    用于同步client-server信息的类
    '''
    def __init__(self, conn: socket.socket, handler, send_back=False):
        '''
        输入:
        conn: 用于通讯的套接字, 要先初始化套接字连接的(ip, port), socket::socket
        handler: 会执行该函数函数, 函数输入为Dict[Any], function
        sendback: 如果开启senback就会在执行完handler后将当前执行的信息发给信息发送方, bool
        '''
        self.conn = conn
        self.handler = handler
        self.send_back = send_back
        self.running = Event()
        self.running.set()

        self.receiver_thread = Thread(target=self._recv_loop, daemon=True)
        self.receiver_thread.start()
        debug_print("BiSocket", "Receiver thread started", "DEBUG")

    def _recv_exact(self, n):
        data = b''
        while len(data) < n:
            try:
                packet = self.conn.recv(n - len(data))
            except Exception as e:
                debug_print("BiSocket", f"Recv error: {e}", "ERROR")
                self.close()
                return None
            if not packet:
                debug_print("BiSocket","Remote side closed connection.", "WARNING")
                self.close()
                return None
            data += packet
        return data

    def _recv_loop(self):
        try:
            debug_print("BiSocket", "Receive loop started, waiting for messages...", "DEBUG")
            while self.running.is_set():
                length_bytes = self._recv_exact(4)
                if length_bytes is None:
                    debug_print("BiSocket", "Failed to receive length bytes", "DEBUG")
                    break

                length = int.from_bytes(length_bytes, 'big')
                debug_print("BiSocket", f"Receiving message of {length} bytes", "DEBUG")
                data = self._recv_exact(length)
                if data is None:
                    debug_print("BiSocket", "Failed to receive data", "DEBUG")
                    break

                debug_print("BiSocket", f"Data received, unpickling...", "DEBUG")
                try:
                    message = pickle.loads(data)
                    debug_print("BiSocket", f"Message unpickled successfully, type: {type(message)}", "DEBUG")
                except Exception as e:
                    debug_print("BiSocket",f"Unpickle error: {e}", "WARNING")
                    import traceback
                    traceback.print_exc()
                    continue

                if self.send_back:
                    try:
                        debug_print("BiSocket", f"Calling handler with message type: {type(message)}", "DEBUG")
                        reply = self.handler(message)
                        debug_print("BiSocket", f"Handler returned, sending reply...", "DEBUG")
                        self.send(reply)
                        debug_print("BiSocket","Sent back response.", "DEBUG")
                    except Exception as e:
                        debug_print("BiSocket",f"Handler/send_back error: {e}", "ERROR")
                        import traceback
                        traceback.print_exc()
                else:
                    try:
                        if message is not None:
                            self.handler(message)
                    except Exception as e:
                        debug_print("BiSocket",f"Handler error: {e}", "ERROR")
        finally:
            debug_print("BiSocket", "Receive loop ending, closing connection...", "DEBUG")
            self.close()

    
    def send(self, data):
        '''
        发送信息:
        data: 发送的信息, Dict[Any]
        '''
        try:
            serialized = pickle.dumps(data)
            debug_print("BiSocket", f"Sending {len(serialized)} bytes...", "DEBUG")
            self.conn.sendall(len(serialized).to_bytes(4, 'big') + serialized)
        except Exception as e:
            debug_print("BiSocket",f"Send failed: {e}", "ERROR")
            self.close()

    def close(self):
        '''
        关闭连接
        '''
        if self.running.is_set():
            self.running.clear()
            try:
                self.conn.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
            self.conn.close()
            debug_print("BiSocket","Connection closed.", "INFO")
