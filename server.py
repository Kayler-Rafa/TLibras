"""
Servidor de desenvolvimento para visualizar palavras reconhecidas pelo TLibras.
Execute este arquivo em um terminal separado antes de iniciar o hand_tracking.py.

Uso:
    python server.py
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import datetime

HOST = "localhost"
PORT = 5000


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            data = json.loads(body)

            palavra = data.get("palavra", "")
            confianca = data.get("confianca", 0)
            letra = data.get("letra", "")
            agora = datetime.datetime.now().strftime("%H:%M:%S")

            print(f"[{agora}]  Letra: {letra!r}  ({confianca:.0%})  →  Palavra: {palavra!r}")

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"ok": true}')
        except Exception as e:
            print(f"[ERRO] {e}")
            self.send_response(400)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # suprime logs HTTP desnecessários


if __name__ == "__main__":
    server = HTTPServer((HOST, PORT), Handler)
    print(f"=== Servidor TLibras iniciado em http://{HOST}:{PORT} ===")
    print("Aguardando reconhecimentos...\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServidor encerrado.")
