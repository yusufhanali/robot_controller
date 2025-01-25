import socket
import os

# Define the socket file
SOCKET_FILE = '/tmp/usta_sockets/gaze.sock'

# Ensure the socket file does not already exist
if os.path.exists(SOCKET_FILE):
    os.remove(SOCKET_FILE)

# Create a Unix domain socket
server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)

# Bind the socket to the file
server.bind(SOCKET_FILE)

# Listen for connections
server.listen(1)
print("Server is listening...")

while True:
    connection, _ = server.accept()
    print("Connection accepted.")
    data = connection.recv(1024)
    print(f"Received: {data.decode('utf-8')}")
    connection.sendall(b"Message received!")
    connection.close()