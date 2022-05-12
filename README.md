# Links
You will find, here, the links to the api and the website.

api : [api link](https://flask-back-app.herokuapp.com/)

website : [Dashboard link](http://54.205.25.37:8501/)

git : [git link](https://github.com/aboustev/OPC-Project_7)

# Dashboard informations
Hosted on AWS on an ec2 instance.

MobaXterm is used to push git modifications on the website/Dashboard and run the website.

# Loading steps of the Dashboard
When you load the Dashboard, it first calls the flask api on its api/load_data function.

Then it loads the informations about the first client and launches api/getdecision/id=100001.

# How the Dashboard is launched using MobaXterm
- Login on the ec2 instance with ubuntu login

- On the bash :

```bash
$ cd OPC-Project_7/08_WEBSITE_AND_MODELS
$ nohup streamlit run streamlit_front.py > log.txt 2>&1 &
```

- Check the [Website](54.205.25.37:8501)

# How the Flask app is saved and launched on heroku
- Followed the steps on this [website](https://dev.to/techparida/how-to-deploy-a-flask-app-on-heroku-heb)
