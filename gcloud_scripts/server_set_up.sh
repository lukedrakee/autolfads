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

# Install MongoDB 6.0 (current stable)
apt-get install -y gnupg curl
curl -fsSL https://pgp.mongodb.com/server-6.0.asc | gpg -o /usr/share/keyrings/mongodb-server-6.0.gpg --dearmor
# Use bookworm for Debian 12
echo 'deb [ signed-by=/usr/share/keyrings/mongodb-server-6.0.gpg ] http://repo.mongodb.org/apt/debian bookworm/mongodb-org/6.0 main' | tee /etc/apt/sources.list.d/mongodb-org-6.0.list
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

# Install Python packages with pip3
apt-get install -y moreutils python3-pip
pip3 install --upgrade pip
pip3 install --upgrade google-cloud-storage
pip3 install --upgrade pymongo
pip3 install grpcio>=1.50.0
pip3 install protobuf>=4.21.0
pip3 install tensorflow>=2.13.0

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

# Configure MongoDB with authentication
gcloud compute ssh ${SERVER_NAME} --zone=${ZONE} --command='sudo mongosh admin --host 127.0.0.1:27017 --eval "db.createUser({user: \"pbt_user\", pwd: \"pbt0Pass\", roles: [ { role: \"userAdminAnyDatabase\", db: \"admin\" }]});db.grantRolesToUser(\"pbt_user\", [{ role: \"readWriteAnyDatabase\", db: \"admin\" }]);" && sudo sed -i "/bindIp/d" /etc/mongod.conf && echo "security:
   authorization: enabled
net:
   bindIp: 127.0.0.1,$(hostname -I)" | sudo tee -a /etc/mongod.conf && sudo systemctl restart mongod.service'
