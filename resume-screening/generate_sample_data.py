"""Generate fictional sample resumes (PDF + TXT) and job descriptions for the demo.

All people, emails and companies below are invented. Run:  python generate_sample_data.py
"""
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

ROOT = Path(__file__).parent / "data"
RES_DIR, JD_DIR = ROOT / "resumes", ROOT / "job_descriptions"

RESUMES = {
    "Elira Hoxha": ("pdf", """elira.hoxha@example.com | +355 69 111 2233 | Tirana, Albania
SUMMARY
Data Scientist with 4 years of experience building machine learning models for retail and fintech. Strong in Python, statistics and communicating insights to stakeholders.
SKILLS
Python, Pandas, NumPy, Scikit-learn, TensorFlow, SQL, PostgreSQL, Machine Learning, Deep Learning, NLP, Data Visualization, Matplotlib, Seaborn, Power BI, Git, Docker, AWS, Statistics, A/B Testing
EXPERIENCE
Data Scientist, FinNova Analytics (2022 - Present)
- Built a credit-risk classification model with XGBoost and Scikit-learn that reduced defaults by 12%.
- Designed A/B tests and statistical analyses for product pricing experiments.
- Deployed models as REST APIs using Flask and Docker on AWS.
Junior Data Analyst, ShopLine Retail (2020 - 2022)
- Wrote SQL queries and Python scripts to automate weekly sales reporting.
- Built Power BI dashboards for sales forecasting and inventory planning.
EDUCATION
MSc Data Science, University of Tirana, 2020
"""),
    "Arben Kola": ("txt", """arben.kola@example.com | +355 68 222 3344
PROFILE
Machine Learning Engineer, 3 years of experience in NLP and computer vision.
TECHNICAL SKILLS
Python, PyTorch, TensorFlow, Keras, Scikit-learn, NLP, spaCy, NLTK, Transformers, Computer Vision, OpenCV, SQL, MongoDB, Docker, Kubernetes, Git, Linux, FastAPI, Machine Learning, Deep Learning
EXPERIENCE
Machine Learning Engineer, VisionTech (2021 - Present)
- Trained transformer models for text classification and named-entity recognition.
- Built a resume parsing pipeline with spaCy and TF-IDF features.
- Served models with FastAPI and Kubernetes.
EDUCATION
BSc Computer Science, Polytechnic University of Tirana, 2021
"""),
    "Sara Mehmeti": ("pdf", """sara.mehmeti@example.com | +355 67 333 4455
SUMMARY
Recent graduate passionate about data analysis and statistics. 1 year of internship experience.
SKILLS
Python, Pandas, Excel, SQL, Tableau, Statistics, Data Visualization, Matplotlib, Communication
EXPERIENCE
Data Analyst Intern, CityBank (2023 - 2024)
- Cleaned customer datasets with Pandas and built Tableau dashboards.
- Performed exploratory data analysis on loan portfolio data.
EDUCATION
BSc Applied Mathematics, University of New York Tirana, 2024
PROJECTS
Student performance predictor using Linear Regression and Random Forest in Scikit-learn.
"""),
    "Dritan Basha": ("pdf", """dritan.basha@example.com | +355 69 444 5566
SUMMARY
Frontend Developer with 5 years of experience building responsive web applications.
SKILLS
JavaScript, TypeScript, React, Next.js, HTML, CSS, Tailwind CSS, Redux, Node.js, REST APIs, GraphQL, Git, Jest, Figma, Agile
EXPERIENCE
Senior Frontend Developer, WebCraft Studio (2021 - Present)
- Led migration of a large dashboard from Angular to React and TypeScript.
- Built a design system with Tailwind CSS and Storybook.
Frontend Developer, PixelSoft (2019 - 2021)
- Developed responsive pages in React and integrated REST APIs.
EDUCATION
BSc Software Engineering, Epoka University, 2019
"""),
    "Megi Dervishi": ("txt", """megi.dervishi@example.com | +355 68 555 6677
SUMMARY
Full Stack Developer with 3 years of experience across React and Django.
SKILLS
Python, Django, JavaScript, React, HTML, CSS, PostgreSQL, Docker, Git, REST APIs, AWS, Agile
EXPERIENCE
Full Stack Developer, CodeBridge (2021 - Present)
- Built REST APIs in Django and frontends in React.
- Set up CI/CD pipelines and Docker deployments on AWS.
EDUCATION
BSc Computer Engineering, Polytechnic University of Tirana, 2021
"""),
    "Klevis Shehu": ("pdf", """klevis.shehu@example.com | +355 67 666 7788
SUMMARY
Digital Marketing Specialist with 6 years of experience in SEO, content and paid campaigns.
SKILLS
SEO, Google Ads, Google Analytics, Content Marketing, Social Media, Copywriting, Email Marketing, Excel, Communication, Leadership
EXPERIENCE
Marketing Lead, BrandUp Agency (2020 - Present)
- Managed a team of 5 and grew organic traffic by 140%.
- Planned Google Ads campaigns with monthly budgets of 50k EUR.
EDUCATION
BA Marketing, University of Tirana, 2018
"""),
    "Anisa Leka": ("pdf", """anisa.leka@example.com | +355 69 777 8899
SUMMARY
Senior Accountant with 7 years of experience in financial reporting and audit.
SKILLS
Accounting, IFRS, Financial Reporting, Excel, SAP, Auditing, Taxation, Budgeting, Communication
EXPERIENCE
Senior Accountant, BalkanAudit (2019 - Present)
- Prepared IFRS financial statements for 20+ clients.
EDUCATION
BSc Finance, University of Tirana, 2017
"""),
    "Ermal Gjoka": ("txt", """ermal.gjoka@example.com | +355 68 888 9900
SUMMARY
Data Engineer with 4 years of experience building data pipelines and warehouses.
SKILLS
Python, SQL, Apache Spark, Airflow, Kafka, AWS, Azure, Snowflake, PostgreSQL, Docker, Git, ETL, Data Warehousing, Linux, Pandas
EXPERIENCE
Data Engineer, StreamFlow (2021 - Present)
- Built Spark and Airflow ETL pipelines processing 2 TB per day.
- Modelled a Snowflake data warehouse used by the analytics team.
Junior Data Engineer, DataHub (2020 - 2021)
EDUCATION
BSc Computer Science, University of Tirana, 2020
"""),
    "Jonida Rama": ("pdf", """jonida.rama@example.com | +355 67 999 0011
SUMMARY
Business Intelligence Analyst with 2 years of experience turning data into dashboards.
SKILLS
SQL, Power BI, Tableau, Excel, Python, Pandas, Statistics, Data Visualization, Communication, Machine Learning
EXPERIENCE
BI Analyst, Telekom Insights (2022 - Present)
- Built Power BI dashboards for customer churn and revenue tracking.
- Wrote SQL and Python scripts to automate KPI reporting.
- Prototyped a churn prediction model with Logistic Regression in Scikit-learn.
EDUCATION
BSc Economics, University of Tirana, 2022
"""),
    "Besnik Cela": ("txt", """besnik.cela@example.com | +355 69 123 4567
SUMMARY
UI/UX Designer with 4 years of experience designing mobile and web products.
SKILLS
Figma, Adobe XD, Sketch, User Research, Wireframing, Prototyping, HTML, CSS, Communication
EXPERIENCE
Product Designer, AppWorks (2021 - Present)
- Designed onboarding flows that improved activation by 18%.
EDUCATION
BA Graphic Design, Academy of Arts Tirana, 2020
"""),
    "Erion Daci": ("pdf", """erion.daci@example.com | +355 68 234 5678
SUMMARY
AI Researcher with 6 years of experience in deep learning and large language models.
SKILLS
Python, PyTorch, TensorFlow, Deep Learning, Machine Learning, NLP, Transformers, Statistics, Scikit-learn, Pandas, NumPy, SQL, Git, Docker, AWS, Data Visualization, Research
EXPERIENCE
Senior AI Researcher, NeuralLab (2020 - Present)
- Fine-tuned transformer models for document classification and semantic search.
- Published 4 papers on NLP and model evaluation.
Data Scientist, Insight Corp (2018 - 2020)
- Built recommendation systems and forecasting models with Scikit-learn.
EDUCATION
PhD Computer Science, University of Bologna, 2018
"""),
    "Fjona Zeqiri": ("txt", """fjona.zeqiri@example.com | +355 67 345 6789
SUMMARY
Backend Developer with 2 years of experience in Java and Spring Boot.
SKILLS
Java, Spring Boot, MySQL, REST APIs, Microservices, Docker, Git, Linux, JUnit, Agile
EXPERIENCE
Backend Developer, EnterpriseSoft (2022 - Present)
- Developed microservices with Spring Boot and MySQL.
EDUCATION
BSc Software Engineering, University of New York Tirana, 2022
"""),
    "Gentian Prifti": ("pdf", """gentian.prifti@example.com | +355 69 456 7890
SUMMARY
Junior React Developer with 1 year of experience and a strong portfolio of web projects.
SKILLS
JavaScript, React, HTML, CSS, Tailwind CSS, Git, REST APIs, Figma, Responsive Design
EXPERIENCE
Junior Frontend Developer, StartHub (2024 - Present)
- Built landing pages and dashboards in React with Tailwind CSS.
EDUCATION
BSc Computer Science, Epoka University, 2024
"""),
    "Liridona Kastrati": ("txt", """liridona.kastrati@example.com | +355 68 567 8901
SUMMARY
HR Generalist with 5 years of experience in recruitment and employee relations.
SKILLS
Recruitment, Onboarding, Employee Relations, Payroll, HR Policies, Communication, Leadership, Excel
EXPERIENCE
HR Generalist, PeopleFirst (2020 - Present)
- Managed end-to-end recruitment for 60+ roles per year.
EDUCATION
BA Psychology, University of Tirana, 2019
"""),
}

JOBS = {
    "data_scientist": """Job Title: Data Scientist
We are hiring a Data Scientist to build machine learning models that drive business decisions.
Requirements:
- 2+ years of experience in data science or machine learning
- Strong Python skills with Pandas, NumPy and Scikit-learn
- Experience with SQL and relational databases such as PostgreSQL
- Knowledge of statistics, A/B testing and machine learning algorithms (regression, classification, clustering)
- Experience with deep learning frameworks such as TensorFlow or PyTorch is a plus
- NLP experience is a plus
- Data visualization with Matplotlib, Seaborn, Power BI or Tableau
- Familiarity with Git, Docker and cloud platforms (AWS)
- Strong communication skills to explain results to non-technical stakeholders
Education: Bachelor's or Master's degree in Computer Science, Statistics, Mathematics or a related field.
""",
    "frontend_developer": """Job Title: Frontend Developer (React)
We are looking for a Frontend Developer to build fast, accessible web applications.
Requirements:
- 2+ years of experience with JavaScript and TypeScript
- Strong knowledge of React and Next.js
- HTML, CSS and Tailwind CSS, responsive design
- State management with Redux
- Consuming REST APIs and GraphQL
- Testing with Jest
- Git and Agile workflows
- Ability to work with designers using Figma
Education: Bachelor's degree in Computer Science, Software Engineering or equivalent experience.
""",
    "data_engineer": """Job Title: Data Engineer
Join our data platform team to design reliable data pipelines.
Requirements:
- 3+ years of experience as a Data Engineer
- Python and advanced SQL
- Apache Spark, Airflow and Kafka
- Cloud data platforms: AWS or Azure, Snowflake
- ETL design and data warehousing
- Docker, Git and Linux
Education: Bachelor's degree in Computer Science or a related field.
""",
}


def write_pdf(path: Path, name: str, body: str) -> None:
    styles = getSampleStyleSheet()
    story = [Paragraph(name, styles["Title"])]
    for line in body.strip().splitlines():
        if line.isupper():
            story += [Spacer(1, 8), Paragraph(line, styles["Heading3"])]
        else:
            story.append(Paragraph(line.replace("&", "&amp;"), styles["Normal"]))
    SimpleDocTemplate(str(path), pagesize=A4, title=f"{name} Resume").build(story)


def main() -> None:
    RES_DIR.mkdir(parents=True, exist_ok=True)
    JD_DIR.mkdir(parents=True, exist_ok=True)
    for name, (fmt, body) in RESUMES.items():
        stem = name.lower().replace(" ", "_")
        if fmt == "pdf":
            write_pdf(RES_DIR / f"{stem}.pdf", name, body)
        else:
            (RES_DIR / f"{stem}.txt").write_text(f"{name}\n{body}", encoding="utf-8")
    for key, text in JOBS.items():
        (JD_DIR / f"{key}.txt").write_text(text, encoding="utf-8")
    print(f"Wrote {len(RESUMES)} resumes and {len(JOBS)} job descriptions to {ROOT}")


if __name__ == "__main__":
    main()
