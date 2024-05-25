# syntax=docker/dockerfile:1

FROM postgres:16
RUN apt-get update && apt-get install -y build-essential git postgresql-server-dev-16 && cd /tmp && git clone --branch v0.6.1 --depth 1 https://github.com/pgvector/pgvector.git && cd pgvector && make && make install && cd .. && rm -rf pgvector && apt-get remove -y build-essential git postgresql-server-dev-16 && apt-get autoremove -y

USER postgres
ENTRYPOINT ["docker-entrypoint.sh"]
EXPOSE 5432
CMD ["postgres"]