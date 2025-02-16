import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


os.environ['OPENAI_API_KEY'] = os.environ.get('OPENAI_API_KEY')



# ### LLM 모델 인스턴스 생성 및 프롬프트 전달 ###

# # LLM 모델 인스턴스 생성
# llm = ChatOpenAI(model="gpt-4o-mini")

# # LLM에 프롬프트 전달
# result = llm.invoke("지구의 자전 주기는?")

# # 결과 출력
# print(result.content)



# ### 프롬프트 템플릿 적용 ###

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



### 순차적인 체인 연결 ###

# 프롬프트 객체 생성 // prompt1: [한국어 단어]를 영어로 번역해줘, prompt2: [영어 단어]를 한국어로 설명해줘
prompt1 = ChatPromptTemplate.from_template("translates {korean_word} to English.")
prompt2 = ChatPromptTemplate.from_template("explain {english_word} using oxford dictionary to me in Korean.")

# LLM 모델 인스턴스 생성
llm = ChatOpenAI(model="gpt-4o-mini")

# chain 1 작업 #
# chain 연결 (LCEL chaining)
chain1 = prompt1 | llm | StrOutputParser()

# chain1 호출
result1 = chain1.invoke({"korean_word":"미래"})

# chain1 결과 출력
print(result1)

# chain2 작업 #
# chain 연결 (LCEL chaining)
chain2 = (
    {"english_word": chain1}
    | prompt2
    | llm
    | StrOutputParser()
)

# chain2 호출
result2 = chain2.invoke({"korean_word":"미래"})

# chain2 결과 출력
print(result2)