#!/bin/bash
# Server Setup Script for PBT - Modernized for current GCP
# Creates a MongoDB server VM with Python 3 and modern dependencies
#
# Usage: sh server_set_up.sh <server-name> <zone>
# Example: sh server_set_up.sh my-server us-central1-a
#
# Note: Use 'sh' to run this script in Google Cloud Shell (not './')

SERVER_NAME=$1
ZONE=$2

if [ -z "$SERVER_NAME" ] || [ -z "$ZONE" ]; then
    echo "Usage: sh server_set_up.sh <server-name> <zone>"
    echo "Example: sh server_set_up.sh my-server us-central1-a"
    exit 1
fi

gcloud config set compute/zone ${ZONE}

# Use standard Debian image (MongoDB server doesn't need ML frameworks)
gcloud compute instances create ${SERVER_NAME} \
    --machine-type=n1-standard-4 \
    --boot-disk-size=300GB \
    --image-project=debian-cloud \
    --image-family=debian-12 \
    --zone=${ZONE} \
    --scopes=cloud-platform \
    --tags=${SERVER_NAME} \
    --metadata startup-script="#!/bin/bash
gcloud config set compute/zone ${ZONE}

# Install basic dependencies first
apt-get update
apt-get install -y git python3 python3-pip python3-full gnupg curl moreutils

# Install MongoDB 7.0 (required for Debian 12/bookworm - MongoDB 6.0 doesn't support bookworm)
curl -fsSL https://www.mongodb.org/static/pgp/server-7.0.asc | gpg -o /usr/share/keyrings/mongodb-server-7.0.gpg --dearmor
echo 'deb [ signed-by=/usr/share/keyrings/mongodb-server-7.0.gpg ] http://repo.mongodb.org/apt/debian bookworm/mongodb-org/7.0 main' | tee /etc/apt/sources.list.d/mongodb-org-7.0.list
apt-get update
apt-get install -y mongodb-org
echo 'mongodb-org hold' | dpkg --set-selections
echo 'mongodb-org-server hold' | dpkg --set-selections
echo 'mongodb-org-shell hold' | dpkg --set-selections
echo 'mongodb-org-mongos hold' | dpkg --set-selections
echo 'mongodb-org-tools hold' | dpkg --set-selections
mkdir -p /db/db_data
systemctl enable mongod.service
systemctl start mongod.service

# Install Python packages with pip3 (--break-system-packages needed for Debian 12 PEP 668)
pip3 install --break-system-packages --upgrade pip
pip3 install --break-system-packages --upgrade google-cloud-storage
pip3 install --break-system-packages --upgrade pymongo
pip3 install --break-system-packages 'grpcio>=1.50.0'
pip3 install --break-system-packages 'protobuf>=4.21.0'
pip3 install --break-system-packages 'tensorflow>=2.13.0'

# Install gcsfuse for bucket mounting
export GCSFUSE_REPO=gcsfuse-\$(lsb_release -c -s)
echo \"deb https://packages.cloud.google.com/apt \$GCSFUSE_REPO main\" | tee /etc/apt/sources.list.d/gcsfuse.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
apt-get update
apt-get install -y gcsfuse

# Enable user_allow_other for FUSE
sed -i 's/#user_allow_other/user_allow_other/' /etc/fuse.conf

echo HOSTNAME=${SERVER_NAME} >> /etc/environment
echo MONGOSERVER=${SERVER_NAME} >> /etc/environment
echo PATH=/code/test-pbt-bucket/code/lfadslite:\$PATH >> /etc/environment
echo PYTHONPATH=/code/test-pbt-bucket/code/lfadslite:/code/test-pbt-bucket/code/PBT_HP_opt/pbt_opt:\$PYTHONPATH >> /etc/environment
"

sleep 40
varxy=$(gcloud compute ssh ${SERVER_NAME} --command="hostname" --zone=${ZONE} 2>/dev/null)
out=$?
sleep_time=40
if [ $out -eq 255 ]; then
    echo "Server boot-up incomplete. Wait for $sleep_time s"
    sleep $sleep_time
elif [ $out -eq 0 ]; then
    echo "Server boot-up completed"
else
    echo "exit status is $out"
fi

# Wait for MongoDB to be installed by startup script (can take several minutes)
echo "Waiting for MongoDB installation to complete (this may take 3-5 minutes)..."
max_attempts=30
attempt=0
while [ $attempt -lt $max_attempts ]; do
    attempt=$((attempt+1))
    echo "Checking for mongosh... (attempt $attempt/$max_attempts)"
    if gcloud compute ssh ${SERVER_NAME} --zone=${ZONE} --command='which mongosh' 2>/dev/null; then
        echo "MongoDB installed successfully!"
        break
    fi
    if [ $attempt -eq $max_attempts ]; then
        echo "ERROR: MongoDB installation timed out. Check startup script logs with:"
        echo "  gcloud compute ssh ${SERVER_NAME} --zone=${ZONE} --command='sudo journalctl -u google-startup-scripts.service'"
        exit 1
    fi
    sleep 10
done

# Wait a bit more for mongod service to start
echo "Waiting for MongoDB service to start..."
sleep 10

# Configure MongoDB with authentication
echo "Configuring MongoDB authentication..."
gcloud compute ssh ${SERVER_NAME} --zone=${ZONE} --command='sudo mongosh admin --host 127.0.0.1:27017 --eval "db.createUser({user: \"pbt_user\", pwd: \"pbt0Pass\", roles: [ { role: \"userAdminAnyDatabase\", db: \"admin\" }]});db.grantRolesToUser(\"pbt_user\", [{ role: \"readWriteAnyDatabase\", db: \"admin\" }]);" && sudo sed -i "/bindIp/d" /etc/mongod.conf && echo "security:
   authorization: enabled
net:
   bindIp: 127.0.0.1,$(hostname -I)" | sudo tee -a /etc/mongod.conf && sudo systemctl restart mongod.service'

echo "Server setup complete!"
