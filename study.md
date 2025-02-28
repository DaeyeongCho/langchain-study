# 랭체인(LangCahin) 입문부터 응용까지

**위키독스 페이지:** 
[랭체인(LangChain) 입문부터 응용까지 위키독스](https://wikidocs.net/book/14473)

**실습 자료 GitHub:**
[tsdata/langchain-study](https://github.com/tsdata/langchain-study)


# 1. LangChain 기초

LangChain의 기본 구성 및 기본 사용법


## 1-1. LangChain이란?

챗봇, 질의응답 시스템, 자동 요약 등 다양한 LLM 애플리케이션을 쉽게 개발할 수 있도록 지원하는 프레임워크

* 2022년 10월 오픈 소스 프로젝트 시작
* 2024년 1월 v0.1.0 공개


### 1-1-1. 랭체인 v0.1.0 출시 의미

랭체인 프로젝트의 첫 번째 안정(stable) 버전

**목표**

* 완전한 하위 호환성 제공

**아키텍처 변경**

1. 'langchain-core' 별도 분리   
'langchain-core'에는 주요 추상화, 인터페이스, 핵심 기능 포함

2. 'langchain'에서 파트너 패키지 분리   
'langchain-communiy', 독립적인 파트너 패키지(langchain-openai 등)를 구분하여 제공

**버전 규칙**

* 마이너 버전(두 번째 숫자): 중대한 변경 발생 시

* 패치 버전(세 번째 숫자): 버그 수정, 새로운 기능 추가


### 1-1-2. 랭체인 프레임워크 구성

* 랭체인 라이브러리(LangChain Libraries): 다양한 컴포넌트의 인터페이스 통합, 컴포넌트들을 체인과 에이전트로 결합할 수 있는 기본 런타임, 체인과 에이전트의 사용을 위한 구현 지원원

* 랭체인 템플릿(LangChain Templates): 다양한 작업을 위한 쉽게 배포할 수 있는 참조 아키텍처 모음

* 랭서크(LangServe): 랭체인 체인의 RESP API 배포 지원

* 랭스미스(LangSmith): 체인 디버깅, 테스트, 평가, 모니터링


### 1-1-3. 필수 라이브러리 설치

랭체인 설치 - langchain-core, langchain-community, langsmith 등 프로젝트 수행에 필수적인 라이브러리 설치

```sh
pip install langchain
```

OpenAI 의존성 패키지 설치 - OpenAI 모델을 사용할 때 필요한 의존성 라이브러리 설치

```sh
# OpenAI의 LLM 모델과 기타 보조 도구 포함
pip install langchain-openai

# OpenAI 모델이 사용하는 토크나이저
pip install tiktoken
```


###	1-1-4. OpenAI 인증키 등록

```py
import os

os.environ['OPENAI_API_KEY'] = os.environ.get('OPENAI_API_KEY')
```


## 1-2. LLM 체인 만들기


### 1-2-1. 기본 LLM 체인(Prompt + LLM)

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


### 1-2-2. 멀티 체인(Multi-Chain)

서로 다른 목적을 가진 체인을 조합하여, 입력 데이터를 다양한 방식으로 처리하고 최종적인 결과를 도출하는 구조   
활용 예시: 복잡한 데이터 처리, 의사 결정, AI 기반 작업 흐름 설계 등


#### 1. 순차적인 체인 연결

2개의 체인(chain1, chain2)를 정의하고, 순차적으로 체인을 연결하는 예제   
chain1: 한국어 단어를 영어로 번역   
chain2: 영어 단어를 한국어로 설명

```py
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

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


### 1-2-3. 체인을 실행하는 방법


#### 1. LangChain의 "Runnable" 프로토콜

"Runnable" 프로토콜: 사용자가 사용자 정의 체인을 쉽게 생성하고 관리할 수 있도록 설계된 핵심적인 개념   
"Runnable" 프로토콜을 통해 일관된 인터페이스를 사용하여 다양한 타입의 컴포넌트를 조합하고, 복잡한 데이터 처리 파이프라인을 구성할 수 있음   

**"Runnable" 프로토콜 주요 메서드**

* invoke: 주어진 입력에 대한 체인 호출 및 결과 반환  
단일 입력에 대해 동기적으로 작동함

* batch: 입력 리스트에 대한 체인 호출, 각 입력에 대한 결과를 리스트로 반환   
여러 입력에 대해 동기적으로 작동, 효율적인 배치 처리가 가능함

* stream: 입력에 대한 체인 호출하고, 결과의 조각들을 스트리밍   
대용량 데이터 처리나 실시간 데이터 처리에 유용함

* 비동기 버전: ainvoke, abatch, astream 등의 메서드는 각각의 동기 버전에 대한 비동기 실행 지원   
이를 통해 비동기 프로그래밍 패러다임을 사용하여 더 높은 처리 성능과 효율을 달성할 수 있음

각 컴포넌트는 입출력 유형이 명확하게 정의되어 있으며, "Runnable" 프로토콜 구현으로 입출력 스키마를 검사할 수 있음   
-> 타입 안정성 보장, 오류 방지

**LangChain을 사용한 커스텀 체인 생성 과정**

1. 필요한 컴포넌트 정의, 각각 "Runnable" 인터페이스 구현

2. 컴포넌트를 조합하여 사용자 정의 체인 생성

3. 생성된 체인을 사용하여 데이터 처리 작업 수행(invoke, batch, stream 메서드 사용)

**예제 코드**

* invoke 예제

```py
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

# 1. 컴포넌트 정의 #
prompt = ChatPromptTemplate.from_template("지구과학에서 {topic}에 대해 간단히 설명해주세요.")
model = ChatOpenAI(model="gpt-4o-mini")
output_parser = StrOutputParser()

# 2. 체인 생성 #
chain = prompt | model | output_parser

# 3-1. invoke 메서드 사용 #
result = chain.invoke({"topic": "지구 자전"})

print("invoke 결과:", result)

# 3-2. batch 메서드 사용 #
# 리스트 입력
topics = ["지구 공전", "화산 활동", "대륙 이동"]
# 각 입력에 대한 결과 체인 호출 및 반환
results = chain.batch([{"topic": t} for t in topics])
for topic, result in zip(topics, results):
    print(f"{topic} 설명: {result[:50]}...")  # 결과의 처음 50자만 출력

# 3-3. stream 메서드 사용 #
# 체인 호출 및 반환
stream = chain.stream({"topic": "지진"})
# 결과 출력
print("stream 결과:")
for chunk in stream:
    print(chunk, end="", flush=True)
print()

# 3-4. 비동기 메서드 사용용 #
# pip install nest_asyncio
import nest_asyncio
import asyncio

# nest_asyncio 적용
nest_asyncio.apply()

# 비동기 메서드 사용 (async/await 구문 필요)
async def run_async():
    result = await chain.ainvoke({"topic": "해류"})
    print("ainvoke 결과:", result[:50], "...")

asyncio.run(run_async())
```


## 1-3. 프롬프트(Prompt)

사용자와 언어 모델 간 대화에서의 입력문   
모델이 제공하는 응답 유형을 결정하는데 중요한 역할을 함


### 1-3-1. 프롬프트 작성 원칙


1. 명확성과 구체성   
모호한 질문은 LLM 모델의 혼란 초래를 방지

2. 배경 정보를 포함   
환각 현상(hallucination) 발생 위험 감소, 관련성 높은 응답 생성

3. 간결함   
핵심 정보에 초점을 두고 불필요한 정보를 배제함   
프롬프트가 길어지면 모델이 덜 중요한 부분에 집중할 수 있음

4. 열린 질문 사용   
자세하고 풍부한 답변을 제공하도록 유도

5. 명확한 목표 설정   
얻고자 하는 정보나 결과의 유형을 정확하게 정의하여 모델이 명확한 지침에 따라 응답을 생성하도록 도움

6. 언어와 문체   
모델이 상황에 맞는 표현을 선택하도록 함


### 1-3-2. 프롬프트 템플릿(PromptTemplate)

단일 문장 또는 명령을 입력하여 프롬프트를 구성할 수 있는 문자열 템플릿


#### 1. 구성요소

프롬프트 구성 시 다양한 구성요소 조합 가능

구성 요소 | 내용
----------|------------------------------------------------------------
지시      | 언어 모델에게 어떤 작업을 수행하도록 요청하는 구체적인 지시 
예시      | 요청된 작업을 수행하는 방법에 대한 하나 이상의 예시
맥락      | 특정 작업을 수행하기 위한 추가적인 맥락
질문      | 어떤 답변을 요구하는 구체적인 질문

#### 2. 문자열 템플릿

프롬프트 템플릿 인스턴스를 생성하고, 실제 입력값을 넣어 프롬프트를 완성하는 예제

```py
from langchain_core.prompts import PromptTemplate

# 1. 'name'과 'age'라는 두 개의 변수를 사용하는 프롬프트 템플릿을 정의
template_text = "안녕하세요, 제 이름은 {name}이고, 나이는 {age}살입니다."

# 2. PromptTemplate 인스턴스를 생성
prompt_template = PromptTemplate.from_template(template_text)

# 3. 템플릿에 값을 채워서 프롬프트를 완성
filled_prompt = prompt_template.format(name="홍길동", age=30)

print(filled_prompt) 
```

#### 3. 프롬프트 템플릿 결합

'+' 연산자를 통한 프롬프트 템플릿 간 결합을 통한 새로운 프롬프트 인스턴스 생성을 지원하는 예제

```py
# 위 예제에 이어서...

# 문자열 템플릿 결합 (PromptTemplate + PromptTemplate + 문자열)
combined_prompt = (
              prompt_template
              + PromptTemplate.from_template("\n\n아버지를 아버지라 부를 수 없습니다.")
              + "\n\n{language}로 번역해주세요."
)

result = combined_prompt.format(name="홍길동", age=30, language="영어")

print(result)
```

체인 생성 및 호출 예제

```py
# 위 예제에 이어서...

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4o-mini")
chain = combined_prompt | llm | StrOutputParser()
result = chain.invoke({"age":30, "language":"영어", "name":"홍길동"})

print(result)
```


### 1-3-3. 챗 프롬프트 템플릿(ChatPromptTemplate)

대화형 상황에서 여러 메시지 입력을 기반으로 단일 메시지 응답 생성에 활용


#### 1. Message 유형

* SystemMessage: 시스템의 기능 설명

* HumanMessage: 사용자 질문

* AIMessage: AI 모델의 응답을 제공합니다.

* FunctionMessage: 특정 함수 호출의 결과를 나타냅니다.

* ToolMessage: 도구 호출의 결과를 나타냅니다.

#### 2. 2-튜플 형태의 메시지 리스트







> 랭체인 관련 정리 추가하기








# 2. RAG(Retrieval-Augmented Generation) 기법

기존 대규모 언어 모델(LLM)을 확장하여, 주어진 질문에 대해 더 정확하고 풍부한 정보를 제공하는 방법   
모델 학습 데이터에 포함되지 않은 외부 데이터를 실시간으로 검색(retrieval)하고, 이를 바탕으로 답변을 생성(generation)하는 과정을 포함함   
환각 현상을 방지하고, 최신 정보를 반영, 더 넓은 지식을 활용할 수 있음

#### 1. RAG 모델의 기본 구조

* 검색 단계(Retrieval Phase): 사용자 질문을 입력으로 하여, 외부 데이터를 검색하는 단계   
검색 엔진, DB 등의 소스에서 정보를 획득함   
질문에 대한 답변을 생성하는데 적합하고 상세한 정보를 포함하는 것이 목표

* 생성 단계(Generation Phase): 검색된 데이터를 기반으로 LLM 모델이 사용자의 질문에 답변하는 단계   
검색된 정보와 기존 지식을 결합하여, 주어진 질문에 대한 답변을 생성함

#### 2. RAG 모델의 장점

* 풍부한 정보 제공: 검색을 통해 획득한 외부 데이터를 활용하여, 구체적이고 풍부한 정보 제공 가능

* 실시간 정보 반영: 최신 데이터를 검색하여, 실시간 데이터 정보에 대응할 수 있음

* 환각 방지: 실제 데이터 기반 답변을 생성함으로써, 환각 현상 발생의 위험을 줄이고 정확도를 높임


## 2-1. RAG 개요

