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

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

# 1. 컴포넌트 정의
prompt = ChatPromptTemplate.from_template("지구과학에서 {topic}에 대해 간단히 설명해주세요.")
model = ChatOpenAI(model="gpt-4o-mini")
output_parser = StrOutputParser()

# 2. 체인 생성
chain = prompt | model | output_parser

# 3. invoke 메소드 사용
result = chain.invoke({"topic": "지구 자전"})

print("invoke 결과:", result)





# batch 메소드 사용
topics = ["지구 공전", "화산 활동", "대륙 이동"]
results = chain.batch([{"topic": t} for t in topics])
for topic, result in zip(topics, results):
    print(f"{topic} 설명: {result[:50]}...")  # 결과의 처음 50자만 출력





# stream 메소드 사용
stream = chain.stream({"topic": "지진"})
print("stream 결과:")
for chunk in stream:
    print(chunk, end="", flush=True)
print()





import nest_asyncio
import asyncio

# nest_asyncio 적용 (구글 코랩 등 주피터 노트북에서 실행 필요)
nest_asyncio.apply()

# 비동기 메소드 사용 (async/await 구문 필요)
async def run_async():
    result = await chain.ainvoke({"topic": "해류"})
    print("ainvoke 결과:", result[:50], "...")

asyncio.run(run_async())