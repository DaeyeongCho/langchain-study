[랭체인(LangChain) 입문부터 응용까지 위키독스](https://wikidocs.net/book/14473)

# 1. LangChain 기초

## LangChain이란?

챗봇, 질의응답 시스템, 자동 요약 등 다양한 LLM 애플리케이션을 쉽게 개발할 수 있도록 지원하는 프레임워크


### 랭체인 v0.1.0 출시 의미

랭체인 프로젝트의 첫 번째 안정(stable) 버전


### 랭체인 프레임워크 구성

* 랭체인 라이브러리(LangChain Libraries)

* 랭체인 템플릿(LangChain Templates)

* 랭서크(LangServe)

* 랭스미스(LangSmith)


### 필수 라이브러리 설치

랭체인 설치 - langchain-core, langchain-community, langsmith 등 프로젝트 수행에 필수적인 라이브러리 설치

```sh
pip install langchain
```

OpenAI 의존성 패키지 설치 - OpenAI 모델을 사용할 때 필요한 의존성 라이브러리 설치

```sh
pip install langchain-openai
pip install tiktoken
```


###	OpenAI 인증키 등록

```py
import os

os.environ['OPENAI_API_KEY'] = os.environ.get('OPENAI_API_KEY')
```


## LLM 체인 만들기

### 기본 LLM 체인(Prompt + LLM)

사용자의 입력(프롬프트)을 받아 LLM을 통해 응답이나 결과를 생성하는 구조


#### 1. 구성 요소

프롬프트(Prompt): 사용자 또는 시스템에서 제공하는 입력으로, LLM에게 특정 작업을 수행하도록 요청하는 지시문
LLM(Large Language Model): GPT, Gemini 등 대규모 언어 모델로, 대량의 텍스트 데이터를 학습하여 언어를 이해하고 생성할 수 있는 인공지능 시스템


#### 2. 일반적인 작동 방식

1. 프롬프트 생성: 사용자의 요구 사항이나 특정 작업을 정의하는 프롬프트를 생성

2. LLM 처리: LLM은 제공된 프롬프트를 분석하고, 학습된 지식을 바탕으로 적절한 응답 생성

3. 응답 반환: LLM에 의해 생성된 응답은 최종 사용자에게 필요한 형태로 변환되어 제공


#### 3. 실습 예제

* LLM 모델 인스턴스 생성 및 프롬프트 전달

```py
from langchain_openai import ChatOpenAI

# LLM 모델 인스턴스 생성
llm = ChatOpenAI(model="gpt-4o-mini")

# LLM에 프롬프트 전달
result = llm.invoke("지구의 자전 주기는?")

# 결과 출력
print(result.content)
```

* 프롬프트 템플릿 적용

```py
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 프롬프트 템플릿 생성
prompt = ChatPromptTemplate.from_template("You are an expert in astronomy. Answer the question. <Question>: {input}")

# 프롬프트 템플릿 생성 결과 출력
print(prompt)

# LLM 모델 인스턴스 생성
llm = ChatOpenAI(model="gpt-4o-mini")

# chain 연결 (LCEL)
chain = prompt | llm

# chain 호출
result = chain.invoke({"input": "지구의 자전 주기는?"})

# 결과 출력
print(result.content)
```

* 문자열 출력 파서(StrOutputParser) 연결, StrOutputParser는 모델의 출력을 문자열 형태로 파싱함

```py
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 프롬프트 객체 생성 // {input} 부분에 사용자 입력이 들어감
prompt = ChatPromptTemplate.from_template("You are an expert in astronomy. Answer the question. <Question>: {input}")

# 프롬프트 객체 생성 결과 출력
print(prompt)


# LLM 모델 인스턴스 생성
llm = ChatOpenAI(model="gpt-4o-mini")

# 출력 파서 생성
output_parser = StrOutputParser()

# chain 연결 (LCEL chaining)
chain = prompt | llm | output_parser

# chain 호출
result = chain.invoke({"input": "지구의 자전 주기는?"})

# 결과 출력 // .content를 붙이지 않음음
print(result)
```


### 멀티 체인(Multi-Chain)

서로 다른 목적을 가진 체인을 조합하여, 입력 데이터를 다양한 방식으로 처리하고 최종적인 결과를 도출하는 구조   
활용 예시: 복잡한 데이터 처리, 의사 결정, AI 기반 작업 흐름 설계 등


#### 1. 순차적인 체인 연결

2개의 체인(chain1, chain2)를 정의하고, 순차적으로 체인을 연결하는 예제   
chain1: 한국어 단어를 영어로 번역   
chain2: 영어 단어를 한국어로 설명

```py
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
```


### 체인을 실행하는 방법

#### 1. LangChain의 "Runnable" 프로토콜

"Runnable" 프로토콜: 사용자가 사용자 정의 체인을 쉽게 생성하고 관리할 수 있도록 설계된 핵심적인 개념   
"Runnable" 프로토콜을 통해 일관된 인터페이스를 사용하여 다양한 타입의 컴포넌트를 조합하고, 복잡한 데이터 처리 파이프라인을 구성할 수 있음

**"Runnable" 프로토콜 주요 메서드**

* invoke: 주어진 입력에 대해 체인을 호출하고, 결과를 반환합니다. 이 메소드는 단일 입력에 대해 동기적으로 작동합니다.

* batch: 입력 리스트에 대해 체인을 호출하고, 각 입력에 대한 결과를 리스트로 반환합니다. 이 메소드는 여러 입력에 대해 동기적으로 작동하며, 효율적인 배치 처리를 가능하게 합니다.

* stream: 입력에 대해 체인을 호출하고, 결과의 조각들을 스트리밍합니다. 이는 대용량 데이터 처리나 실시간 데이터 처리에 유용합니다.

* 비동기 버전: ainvoke, abatch, astream 등의 메소드는 각각의 동기 버전에 대한 비동기 실행을 지원합니다. 이를 통해 비동기 프로그래밍 패러다임을 사용하여 더 높은 처리 성능과 효율을 달성할 수 있습니다.

각 컴포넌트는 입력 및 출력 유형이 명확하게 정의되어 있음