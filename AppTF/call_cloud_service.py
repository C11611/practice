from oauth2client.client import GoogleCredentials
import googleapiclient.discovery

#change this values to match your project
PROJECT_ID=""  #google云上的项目ID
MODEL_NAME=""  #已经部署在云服务上的模型名称
CREDENTIALS_FILE=""  #google 云上下载的文件

#These are the values we want a prediction for  #真实情况下一般需要读取数据进行传入
inputs_for_prediction=[
    {"input":[0.4999,1.0,0.0,1.0,0.0,0.0,0.0,0.5]}
]

#connect to the google cloud-ML service
credentials=GoogleCredentials.from_stream(CREDENTIALS_FILE)
service=googleapiclient.discovery.build('ml','v1',credentials=credentials)

#connect to our prediction model
name='projects/{}/models/{}'.format(PROJECT_ID,MODEL_NAME)
response=service.projects().predict(
    name=name,
    body={'instances':inputs_for_prediction}
).execute

#report any errors
if  'error' in response:
    raise RuntimeError(response['error'])

#grab the results from the response object
results=response['predictions']

#print the results!
print(results)