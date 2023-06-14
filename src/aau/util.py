userid="group99999"

sensors = ["temp", "light"]
topic = "ubiss/"
server="130.225.57.224"
#server ="192.168.10.1"
#fileserver="192.168.10.1"


#possible_topics = []
# for s in sensors:
#     possible_topics.append(s)


def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))

# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, msg):
    print(msg.topic+" "+str(msg.payload))

def serialize(payload_list,timestamps,topic):
    string_out=str(topic)
    print(payload_list)
    for entry in payload_list:
        string_out = string_out +"," + str(entry)
    string_out+= ',ts'
    for entry in timestamps:
        string_out = string_out +"," + str(entry)
    return string_out


def send_topics(topic_payload,userid,client):
    output = str(userid)
    if len(topic_payload['topic'])== 1:
        full_topic= topic+topic_payload['topic'][0]
    else:
        full_topic=topic+"multiple"

    for i in range(len(topic_payload["topic"])):
        payload= topic_payload['payload'][i]
        timestamp = topic_payload['ts'][i]

        serialized_input = serialize(payload,timestamp,topic_payload['topic'][i])
        output = output + "," + serialized_input
         
    client.publish(full_topic, output)

    return



def prepare_payload(sensors, data, timestamps):
#Specific requirements is that the "data", must be int or float,
# The timestamps must contain ":", and this should not be used in the sensors field.
# Wheter hour:minute:second is sent, or day/hour:minute:second, or minute:second, it will work
# if a ":" is present.
# The "sensors", must be a string, the name does not matter as it is more of an informative 
# parameter for you, so you could map "t" to temperature, but this is a matter for you to control.   
    if len(sensors)==1:
        payload = {"topic":[], "payload":[], "ts":[]}
        for s in sensors:
            payload["topic"].append(s)
        for d in data:
            flag = True
            try:
              # try converting to integer
              float(d)
            except ValueError:
              flag = False
            if flag==False:
                print("sensor value must be a number, int or float. No strings allowed")
                return -1
            payload["payload"].append(d)
        for t in timestamps:
            payload["ts"].append(t)
        return payload

    elif len(sensors)!= len(data) and len(sensors)!=len(timestamps):
        print("Must have equal set length of sensors, payload and timestamps")
        return -1
    payload = {"topic":[], "payload":[], "ts":[]}
    for s in sensors:
        payload["topic"].append(s)
    for d in data:
        payload["payload"].append(d)
    for t in timestamps:
        payload["ts"].append(t)
    print(payload["ts"])
    print(payload["payload"])
    return payload
    
    
