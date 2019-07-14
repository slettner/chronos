cp Dockerfile ../../.
cd ../../.
docker build --rm=true -t ga63fiy/chronos/0.0.1-gpu .
docker push ga63fiy/chronos/0.0.1-gpu
rm Dockerfile