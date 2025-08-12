pip install flask
pip install numpy
brew install cmake
pip install face_recognition
pip install qrcode

mkdir src/certs -p
cd src/certs
openssl req -x509 -newkey rsa:2048 -nodes \
  -keyout key.pem \
  -out cert.pem \
  -days 365
