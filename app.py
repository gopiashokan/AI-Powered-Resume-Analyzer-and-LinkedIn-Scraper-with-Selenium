import time
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import warnings
warnings.filterwarnings('ignore')


def streamlit_config():

    # page configuration
    st.set_page_config(page_title='Resume Analyzer AI', layout="wide")

    # page header transparent color
    page_background_color = """
    <style>

    [data-testid="stHeader"] 
    {
    background: rgba(0,0,0,0);
    }

    </style>
    """
    st.markdown(page_background_color, unsafe_allow_html=True)

    # title and position
    st.markdown(f'<h1 style="text-align: center;">Resume Analyzer AI</h1>',
                unsafe_allow_html=True)


def pdf_to_chunks(pdf):
    # read pdf and it returns memory address
    pdf_reader = PdfReader(pdf)

    # extrat text from one by one page
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # Set a really small chunk size, just to show.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=200,
        length_function=len)

    chunks = text_splitter.split_text(text=text)
    return chunks


def resume_summary(query_with_chunks):
    chunks = f''' need to detailed summarization of below resume and finally conclude them

                """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                {query_with_chunks}
                """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                '''
    return chunks


def resume_strength(query_with_chunks):
    chunks = f'''need to detailed analysis and explain of the strength of below resume and finally conclude them
                """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                {query_with_chunks}
                """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                '''
    return chunks


def resume_weakness(query_with_chunks):
    chunks = f'''need to detailed analysis and explain of the weakness of below resume details and how to improve make a better resume

                """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                {query_with_chunks}
                """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                '''
    return chunks


def job_title_suggestion(query_with_chunks):

    chunks = f''' what are the job roles i apply to likedin based on below?
                  
                  """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                  {query_with_chunks}
                  """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                '''
    return chunks


def openai(openai_api_key, chunks):

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstores = FAISS.from_texts(chunks, embedding=embeddings)

    docs = vectorstores.similarity_search(query=chunks, k=3)

    llm = OpenAI(model_name='gpt-3.5-turbo', openai_api_key=openai_api_key)
    chain = load_qa_chain(llm=llm, chain_type='stuff')
    response = chain.run(input_documents=docs, question=chunks)
    return response


class linkedin_scrap:

    def linkedin_open_scrolldown(driver, user_job_title):

        b = []
        for i in user_job_title:
            x = i.split()
            y = '%20'.join(x)
            b.append(y)
        job_title = '%2C%20'.join(b)

        link = f"https://in.linkedin.com/jobs/search?keywords={job_title}&location=India&locationId=&geoId=102713980&f_TPR=r604800&position=1&pageNum=0"

        driver.get(link)
        driver.implicitly_wait(10)

        c = 0
        while c < 5:
            driver.execute_script(
                "window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(5)
            c += 1
            try:
                x = driver.find_element(
                    by=By.XPATH, value="//button[@aria-label='See more jobs']")
                driver.execute_script('arguments[0].click();', x)
                time.sleep(3)
            except:
                pass
                time.sleep(4)

    def job_title_filter(x, user_job_title):

        s = [i.lower() for i in user_job_title]
        suggestion = []
        for i in s:
            suggestion.extend(i.split())

        s = x.split()
        a = [i.lower() for i in s]

        intersection = list(set(suggestion).intersection(set(a)))
        return x if len(intersection) > 1 else np.nan

    def get_description(driver, link):
        driver.get(link)
        driver.find_element(
            by=By.CLASS_NAME, value='show-more-less-html__button').click()
        description = driver.find_elements(
            by=By.CLASS_NAME, value='show-more-less-html')
        driver.implicitly_wait(4)

        for j in description:
            text = j.text.replace('Show less', '')
            return text.strip()

    def data_scrap(driver, user_job_title):

        company_name = []
        try:
            company = driver.find_elements(
                by=By.CLASS_NAME, value='base-search-card__subtitle')
            for i in company:
                company_name.append(i.text)
        except:
            pass

        job_title = []
        try:
            title = driver.find_elements(
                by=By.CLASS_NAME, value='base-search-card__title')
            for i in title:
                job_title.append(i.text)
        except:
            pass

        company_location = []
        try:
            location = driver.find_elements(
                by=By.CLASS_NAME, value='job-search-card__location')
            for i in location:
                company_location.append(i.text)
        except:
            pass

        job_url = []
        try:
            url = driver.find_elements(
                by=By.XPATH, value="//a[contains(@href, '/jobs/')]")
            url_list = [i.get_attribute('href') for i in url]
            for url in url_list:
                job_url.append(url)
        except:
            pass

        # combine the all data to single dataframe
        df = pd.DataFrame(company_name, columns=['Company Name'])
        df['Job Title'] = pd.DataFrame(job_title)
        df['Location'] = pd.DataFrame(company_location)
        df['Website URL'] = pd.DataFrame(job_url)

        # job title filter based on chatgpt suggestion
        df['Job Title'] = df['Job Title'].apply(
            lambda x: linkedin_scrap.job_title_filter(x, user_job_title))
        df = df.dropna()
        df.reset_index(drop=True, inplace=True)
        df = df.iloc[:30, :]

        # list of url after filter df
        website_url = df['Website URL'].tolist()

        # add job description in df
        job_description = []
        for i in range(0, len(website_url)):
            link = website_url[i]
            time.sleep(5)
            data = linkedin_scrap.get_description(driver, link)
            if data is not None and len(data.strip()) > 0:
                job_description.append(data)
            else:
                job_description.append('Description Not Available')

        df['Job Description'] = pd.DataFrame(
            job_description, columns=['Description'])
        df = df.dropna()
        df.reset_index(drop=True, inplace=True)
        return df

    def main(user_job_title):
        # chromedriver setup
        options = Options()
        options.add_argument("--start-maximized")
        driver = webdriver.Chrome('https://github.com/gopiashokan/Resume-Analyzer-Artificial-Intelligence/blob/main/chromedriver.exe', options=options)

        linkedin_scrap.linkedin_open_scrolldown(driver, user_job_title)

        final_df = linkedin_scrap.data_scrap(driver, user_job_title)
        return final_df


streamlit_config()
st.write('')


# file upload
pdf = st.file_uploader(label='', type='pdf')
openai_api_key = st.text_input(label='OpenAI API Key', type='password')


# sidebar
with st.sidebar:

    add_vertical_space(3)

    option = option_menu(menu_title='', options=['Summary', 'Strength', 'Weakness', 'Suggestion', 'Linkedin Jobs', 'Exit'],
                         icons=['house-fill', 'database-fill', 'pass-fill', 'list-ul', 'linkedin', 'sign-turn-right-fill'])



if option == 'Summary':

    try:
        if pdf is not None and openai_api_key is not None:
            pdf_chunks = pdf_to_chunks(pdf)

            summary = resume_summary(query_with_chunks=pdf_chunks)
            result_summary = openai(openai_api_key=openai_api_key, chunks=summary)

            st.subheader('Summary:')
            st.write(result_summary)

    except:
        col1, col2 = st.columns(2)
        with col1:
            st.warning('OpenAI API Key is Invalid')


elif option == 'Strength':

    try:
        if pdf is not None and openai_api_key is not None:

            pdf_chunks = pdf_to_chunks(pdf)

            # Resume summary
            summary = resume_summary(query_with_chunks=pdf_chunks)
            result_summary = openai(
                openai_api_key=openai_api_key, chunks=summary)

            strength = resume_strength(query_with_chunks=result_summary)
            result_strength = openai(
                openai_api_key=openai_api_key, chunks=strength)

            st.subheader('Strength:')
            st.write(result_strength)

    except:
        col1, col2 = st.columns(2)
        with col1:
            st.warning('OpenAI API Key is Invalid')


elif option == 'Weakness':

    try:
        if pdf is not None and openai_api_key is not None:

            pdf_chunks = pdf_to_chunks(pdf)

            # Resume summary
            summary = resume_summary(query_with_chunks=pdf_chunks)
            result_summary = openai(
                openai_api_key=openai_api_key, chunks=summary)

            weakness = resume_weakness(query_with_chunks=result_summary)
            result_weakness = openai(
                openai_api_key=openai_api_key, chunks=weakness)

            st.subheader('Weakness:')
            st.write(result_weakness)

    except:
        col1, col2 = st.columns(2)
        with col1:
            st.warning('OpenAI API Key is Invalid')


elif option == 'Suggestion':

    try:
        if pdf is not None and openai_api_key is not None:
            pdf_chunks = pdf_to_chunks(pdf)

            # Resume summary
            summary = resume_summary(query_with_chunks=pdf_chunks)
            result_summary = openai(
                openai_api_key=openai_api_key, chunks=summary)

            job_suggestion = job_title_suggestion(
                query_with_chunks=result_summary)
            result_suggestion = openai(
                openai_api_key=openai_api_key, chunks=job_suggestion)

            st.subheader('Suggestion: ')
            st.write(result_suggestion)

    except:
        col1, col2 = st.columns(2)
        with col1:
            st.warning('OpenAI API Key is Invalid')


elif option == 'Linkedin Jobs':

    if pdf is not None and openai_api_key is not None:

        

            # get user input of job title
            user_input_job_title = st.text_input(
                    label='Enter Job Titles (with comma separated):')
            submit = st.button('Submit')
                
            if submit and len(user_input_job_title)>0:
                user_job_title = user_input_job_title.split(',')

                df = linkedin_scrap.main(user_job_title)

                l = len(df['Company Name'])
                for i in range(0, l):
                    st.write(f"Company Name : {df.iloc[i,0]}")
                    st.write(f"Job Title    : {df.iloc[i,1]}")
                    st.write(f"Location     : {df.iloc[i,2]}")
                    st.write(f"Website URL  : {df.iloc[i,3]}")
                    with st.expander(label='Job Desription'):
                            st.write(df.iloc[i, 4])
                    st.write('')
                    st.write('')

            else:
                col1, col2 = st.columns(2)
                with col1:
                    st.info('Please Enter the Job Titles')
    
            

elif option == 'Exit':

    st.success('Thank you for your time. Exiting the application')
    st.balloons()

