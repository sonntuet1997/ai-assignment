docker-compose -f docker-compose-production.yml kill
docker image rm $(docker-compose -f docker-compose-production.yml images -q) -f
docker image prune -f
docker-compose -f docker-compose-production.yml up -d --build --force-recreate -V
