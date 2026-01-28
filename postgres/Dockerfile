FROM postgres:16-alpine

# Set default environment variables
ENV POSTGRES_USER=postgres
ENV POSTGRES_PASSWORD=admin
ENV POSTGRES_DB=qt328pp

# Copy init scripts (if any) to run on first startup
COPY init.sql /docker-entrypoint-initdb.d/

EXPOSE 5432
