import os

os.environ['OPENAI_API_KEY'] = os.environ.get('OPENAI_API_KEY')



# ### LLM 모델 인스턴스 생성 및 프롬프트 전달 ###

# from langchain_openai import ChatOpenAI

# # LLM 모델 인스턴스 생성
# llm = ChatOpenAI(model="gpt-4o-mini")

# # LLM에 프롬프트 전달
# result = llm.invoke("지구의 자전 주기는?")

# # 결과 출력
# print(result.content)



# ### 프롬프트 템플릿 적용 ###

# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser

# # 프롬프트 객체 생성 // {input} 부분에 사용자 입력이 들어감
# prompt = ChatPromptTemplate.from_template("You are an expert in astronomy. Answer the question. <Question>: {input}")

# # 프롬프트 객체 생성 결과 출력
# print(prompt)


# # LLM 모델 인스턴스 생성
# llm = ChatOpenAI(model="gpt-4o-mini")

# # 출력 파서 생성
# output_parser = StrOutputParser()

# # chain 연결 (LCEL chaining)
# chain = prompt | llm | output_parser

# # chain 호출
# result = chain.invoke({"input": "지구의 자전 주기는?"})

# # 결과 출력 // .content를 붙이지 않음
# print(result)



# ### 순차적인 체인 연결 ###

# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser

# # 프롬프트 객체 생성 // prompt1: [한국어 단어]를 영어로 번역해줘, prompt2: [영어 단어]를 한국어로 설명해줘
# prompt1 = ChatPromptTemplate.from_template("translates {korean_word} to English.")
# prompt2 = ChatPromptTemplate.from_template("explain {english_word} using oxford dictionary to me in Korean.")

# # LLM 모델 인스턴스 생성
# llm = ChatOpenAI(model="gpt-4o-mini")

# # chain 1 작업 #
# # chain 연결 (LCEL chaining)
# chain1 = prompt1 | llm | StrOutputParser()

# # chain1 호출
# result1 = chain1.invoke({"korean_word":"미래"})

# # chain1 결과 출력
# print(result1)

# # chain2 작업 #
# # chain 연결 (LCEL chaining)
# chain2 = (
#     {"english_word": chain1}
#     | prompt2
#     | llm
#     | StrOutputParser()
# )

# # chain2 호출
# result2 = chain2.invoke({"korean_word":"미래"})

# # chain2 결과 출력
# print(result2)

# from langchain_openai import ChatOpenAI
# from langchain.prompts import ChatPromptTemplate
# from langchain.schema.output_parser import StrOutputParser

# # 1. 컴포넌트 정의
# prompt = ChatPromptTemplate.from_template("지구과학에서 {topic}에 대해 간단히 설명해주세요.")
# model = ChatOpenAI(model="gpt-4o-mini")
# output_parser = StrOutputParser()

# # 2. 체인 생성
# chain = prompt | model | output_parser

# # 3. invoke 메소드 사용
# result = chain.invoke({"topic": "지구 자전"})

# print("invoke 결과:", result)





# # batch 메소드 사용
# topics = ["지구 공전", "화산 활동", "대륙 이동"]
# results = chain.batch([{"topic": t} for t in topics])
# for topic, result in zip(topics, results):
#     print(f"{topic} 설명: {result[:50]}...")  # 결과의 처음 50자만 출력





# # stream 메소드 사용
# stream = chain.stream({"topic": "지진"})
# print("stream 결과:")
# for chunk in stream:
#     print(chunk, end="", flush=True)
# print()





# import nest_asyncio
# import asyncio

# # nest_asyncio 적용 (구글 코랩 등 주피터 노트북에서 실행 필요)
# nest_asyncio.apply()

# # 비동기 메소드 사용 (async/await 구문 필요)
# async def run_async():
#     result = await chain.ainvoke({"topic": "해류"})
#     print("ainvoke 결과:", result[:50], "...")

# asyncio.run(run_async())





# from langchain_core.prompts import PromptTemplate

# # 'name'과 'age'라는 두 개의 변수를 사용하는 프롬프트 템플릿을 정의
# template_text = "안녕하세요, 제 이름은 {name}이고, 나이는 {age}살입니다."

# # PromptTemplate 인스턴스를 생성
# prompt_template = PromptTemplate.from_template(template_text)

# # 템플릿에 값을 채워서 프롬프트를 완성
# filled_prompt = prompt_template.format(name="홍길동", age=30)

# print(filled_prompt)





# # 문자열 템플릿 결합 (PromptTemplate + PromptTemplate + 문자열)
# combined_prompt = (
#               prompt_template
#               + PromptTemplate.from_template("\n\n아버지를 아버지라 부를 수 없습니다.")
#               + "\n\n{language}로 번역해주세요."
# )

# print(combined_prompt)


# result = combined_prompt.format(name="홍길동", age=30, language="영어")

# print(result)

# from langchain_openai import ChatOpenAI
# from langchain_core.output_parsers import StrOutputParser

# llm = ChatOpenAI(model="gpt-4o-mini")
# chain = combined_prompt | llm | StrOutputParser()
# result = chain.invoke({"age":30, "language":"영어", "name":"홍길동"})

# print(result)




# # 2-튜플 형태의 메시지 목록으로 프롬프트 생성 (type, content)

# from langchain_core.prompts import ChatPromptTemplate

# chat_prompt = ChatPromptTemplate.from_messages([
#     ("system", "이 시스템은 천문학 질문에 답변할 수 있습니다."),
#     ("user", "{user_input}"),
# ])

# messages = chat_prompt.format_messages(user_input="태양계에서 가장 큰 행성은 무엇인가요?")
# print(messages)




# from langchain_openai import ChatOpenAI
# from langchain_core.output_parsers import StrOutputParser

# llm = ChatOpenAI(model="gpt-4o-mini")

# chain = chat_prompt | llm | StrOutputParser()

# result = chain.invoke({"user_input": "태양계에서 가장 큰 행성은 무엇인가요?"})

# print(result)




# # MessagePromptTemplate 활용

# from langchain_core.prompts import SystemMessagePromptTemplate,  HumanMessagePromptTemplate

# chat_prompt = ChatPromptTemplate.from_messages(
#     [
#         SystemMessagePromptTemplate.from_template("이 시스템은 천문학 질문에 답변할 수 있습니다."),
#         HumanMessagePromptTemplate.from_template("{user_input}"),
#     ]
# )

# messages = chat_prompt.format_messages(user_input="태양계에서 가장 큰 행성은 무엇인가요?")


# print(messages)




# Data Loader - 웹페이지 데이터 가져오기
from langchain_community.document_loaders import WebBaseLoader

# 위키피디아 정책과 지침
url = 'https://ko.wikipedia.org/wiki/%EC%9C%84%ED%82%A4%EB%B0%B1%EA%B3%BC:%EC%A0%95%EC%B1%85%EA%B3%BC_%EC%A7%80%EC%B9%A8'
loader = WebBaseLoader(url)

# 웹페이지 텍스트 -> Documents
docs = loader.load()

print(len(docs))
print(len(docs[0].page_content))
print(docs[0].page_content[5000:6000])


# Text Split (Documents -> small chunks: Documents)
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

print(len(splits))
print(splits[10])

print()

print(splits[10].page_content)

print()

print(splits[10].metadata)


# Indexing (Texts -> Embedding -> Store)
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

vectorstore = Chroma.from_documents(documents=splits,
                                    embedding=OpenAIEmbeddings())

docs = vectorstore.similarity_search("격하 과정에 대해서 설명해주세요.")
print(len(docs))
print(docs[0].page_content)



from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Prompt
template = '''Answer the question based only on the following context:
{context}

Question: {question}
'''

prompt = ChatPromptTemplate.from_template(template)

# LLM
model = ChatOpenAI(model='gpt-4o-mini', temperature=0)

# Rretriever
retriever = vectorstore.as_retriever()

# Combine Documents
def format_docs(docs):
    return '\n\n'.join(doc.page_content for doc in docs)

# RAG Chain 연결
rag_chain = (
    {'context': retriever | format_docs, 'question': RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

# Chain 실행
result = rag_chain.invoke("격하 과정에 대해서 설명해주세요.")

print(result)