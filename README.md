# Site for recognizing
- Python
- Django
- PostdreSQL
- Docker
- SonarQube

### Getting Started
##### To run project locally, follow next steps:

Clone the repository: ``` git clone https://github.com/MariaHusak/Site_for_recognizing2.git ```

Create a ```.env``` file in the root directory, for example:

```
SECRET_KEY =your_django_secret_key
EMAIL_HOST_PASSWORD =your_email_password
EMAIL_HOST_USER =your_email
DEFAULT_FROM_EMAIL =your_email

DB_HOST=db
DB_NAME=dbname
DB_USER=dbuser
DB_PASS=dbpassword

SONAR_HOST_URL="http://sonarqube:9000"
SONAR_PROJECT_KEY=your_project_key
SONAR_PROJECT_NAME=your_project_name
SONAR_TOKEN=your_project_token

SONAR_JDBC_URL="jdbc:postgresql://db:5432/sonar"
SONAR_JDBC_USERNAME=sonar_name
SONAR_JDBC_PASSWORD=sonar_password

```
Run project in the root directory, using command: ``` docker-compose up --build ```

Open links in browser:

Application: http://localhost:8000

SonarQube: http://localhost:9000

### Project Structure

```
Site_for_recognizing2/
├── Dockerfile
├── docker-compose.yml
├── manage.py
├── requirements.txt
├── .dockerignore
├── .gitignore
├── recognition/
│   ├── model/
│   ├── recognition/
│   ├── scripts/
│   ├── staticfiles/
│   │   ├── admin/
│   │   ├── css/
│   │   ├── img/
│   │   ├── js/
│   ├── uploading_data/
│   └─── user/
└── README.md
```

### Architecture Diagram

![Architecture](images/architecture.png)
