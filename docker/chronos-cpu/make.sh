cp Dockerfile ../../.
cd ../../.
docker build --rm=true -t ga63fiy/chronos:0.0.1 .
# docker push ga63fiy/chronos:0.0.1
rm Dockerfile