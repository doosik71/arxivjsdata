# Audiopedia: Audio QA with Knowledge

Abhirama Subramanyam Penamakuri, Kiran Chhatre, Akshat Jain (2024)

## 🧩 Problem to Solve

기존의 Audio Question Answering (AQA) 연구들은 주로 입력된 오디오 신호 자체에서 정답을 찾을 수 있는 단순한 질의에 집중해 왔다. 그러나 실제 환경에서는 오디오에 언급된 특정 명명 엔티티(Named Entity)에 대해 외부 세계 지식을 결합하여 추론해야 하는 상황이 빈번하게 발생한다. 예를 들어, 오디오에서 "KFC에서 밥을 먹었다"라는 말이 나왔을 때, "그 식당은 어느 나라에서 설립되었는가?"라는 질문에 답하기 위해서는 오디오 외부의 지식 베이스(Knowledge Base)를 통한 추론이 필수적이다.

본 논문의 목표는 이러한 지식 집약적(Knowledge-intensive) 오디오 질의응답 문제를 정의하고, 이를 벤치마킹하기 위한 데이터셋인 **Audiopedia**를 제안하는 것이다. 또한, 기존의 Large Audio Language Models (LALMs)가 이러한 지식 기반 추론에 취약하다는 점을 해결하기 위해, 외부 지식을 모델에 주입할 수 있는 범용 프레임워크를 제시한다.

## ✨ Key Contributions

본 논문의 핵심 기여는 오디오 이해와 외부 지식 추론을 동시에 요구하는 새로운 태스크인 **Audiopedia**를 정의하고, 이를 해결하기 위한 **AEL (Audio Entity Linking)** 및 **$KA^2LM$ (Knowledge-Augmented Audio Large Multimodal Model)** 프레임워크를 제안한 것이다.

핵심 설계 아이디어는 LALM이 모든 외부 지식을 내재적으로 학습하도록 하는 대신, 오디오에서 언급된 엔티티를 지식 베이스에서 찾아내어 해당 정보를 텍스트 형태로 프롬프트에 주입하는 '플러그 앤 플레이' 방식의 증강 구조를 채택한 것이다.

## 📎 Related Works

질의응답(QA) 연구는 입력 모달리티에 따라 Textual QA (TQA), Visual QA (VQA), Audio QA (AQA), Video QA (VideoQA)로 구분된다. 최근 LLM의 발전으로 입력 콘텐츠 내부의 정보만으로 답하는 것을 넘어 외부 지식을 활용하는 Knowledge-aware 벤치마크가 TQA와 VQA 분야에서는 활발히 연구되어 왔다 (예: KVQA, A-OKVQA).

그러나 오디오 분야의 AQA는 상대적으로 최신 연구 영역이며, 외부 지식을 요구하는 지식 집약적 벤치마크가 부재한 상태였다. 본 논문은 이러한 공백을 메우기 위해 VQA의 지식 기반 접근 방식을 오디오 도메인으로 확장하여 제안한다.

## 🛠️ Methodology

### 1. Audiopedia 벤치마크 구성

본 논문은 난이도와 요구 사항에 따라 세 가지 서브 태스크를 정의한다.

- **s-AQA (Single Audio QA):** 단일 오디오 샘플 내의 엔티티를 기반으로 외부 지식을 이용해 정답을 도출한다.
- **m-AQA (Multi-Audio QA):** 여러 개의 오디오 샘플에 걸쳐 언급된 엔티티들의 지식을 종합하여 추론한다 (예: "이 식당들은 모두 같은 시대에 설립되었는가?").
- **r-AQA (Retrieval-Augmented Audio QA):** 주어진 오디오 풀(pool)에서 질문과 관련 있는 오디오를 먼저 검색(Retrieval)한 후, 그 내용을 바탕으로 지식 기반 추론을 수행한다.

데이터셋은 TextKVQA의 Business-KB(Wikidata 기반)를 활용하여 합성되었으며, Tachotron 2 TTS 모델을 통해 텍스트를 오디오로 변환하여 구축하였다.

### 2. Audio Entity Linking (AEL)

AEL은 오디오 내의 명명 엔티티를 식별하고 외부 지식 베이스의 엔티티와 연결하는 과정이다.

- **지식 인코딩:** 지식 베이스의 각 엔티티 $N_i$에 대한 트리플렛(triplet) 정보를 문장으로 변환하고, 텍스트 인코더를 통해 임베딩 $E^T_{\{1,2,..n\}}$를 생성한다.
- **오디오 인코딩:** 입력 오디오 $A$를 `wav2vec 2.0`으로 전사(transcribe)한 후, 동일한 텍스트 인코더로 임베딩 $A^T$를 생성한다.
- **엔티티 연결:** $A^T$와 모든 $E^T$ 간의 코사인 유사도를 계산하여 가장 유사도가 높은 엔티티의 지식 $K_l$을 선택한다.

### 3. Knowledge-Augmented Audio Large Multimodal Model ($KA^2LM$)

$KA^2LM$은 AEL을 통해 획득한 외부 지식을 LALM의 입력 프롬프트에 주입하는 범용 프레임워크이다.

- **프롬프트 증강:** 일반적인 LALM은 $\text{Instruction} + \text{Audio} + \text{Question}$을 입력으로 받지만, $KA^2LM$은 여기에 AEL로 추출된 지식 $K_l$을 추가하여 $\text{Instruction} + K_l + \text{Audio} + \text{Question}$ 형태로 구성한다.
- 이를 통해 모델은 외부 지식 베이스의 구체적인 정보를 참고하여 정답 $a$를 생성할 수 있다.

### 4. 태스크별 절차

- **s-AQA:** $\text{Audio} \rightarrow \text{AEL} \rightarrow K_l \rightarrow KA^2LM \rightarrow \text{Answer}$
- **m-AQA:** $\text{Multiple Audios} \rightarrow \text{AEL per audio} \rightarrow \text{Concatenated } K_l \rightarrow KA^2LM \rightarrow \text{Answer}$
- **r-AQA:** $\text{Audio Pool} \rightarrow \text{Cosine Similarity (Question vs Audios)} \rightarrow \text{Filter by threshold } (t=0.25) \rightarrow \text{Proceed as m-AQA}$

## 📊 Results

### 1. 실험 설정

- **평가 모델:** Audio-flamingo, GAMA, LTU-AS 세 가지 LALM을 사용하였다.
- **지표:** s-AQA와 m-AQA는 Accuracy를, r-AQA는 F1 score를 사용하여 측정하였다.
- **설정:** 모든 실험은 파인튜닝 없이 Zero-shot 설정에서 진행되었다.

### 2. 주요 결과

- **지식 증강의 효과:** 모든 모델에서 $KA^2LM$을 적용했을 때 성능이 비약적으로 상승하였다. 특히 open-ended 답변이 필요한 s-AQA에서 그 효과가 두드러졌다.
- **모델별 성능:** LTU-AS가 가장 우수한 성능을 보였으며, $KA^2LM$ 적용 시 s-AQA에서 54.7%의 정확도를 기록하였다.
- **지식 양의 영향:** 엔티티 이름만 제공했을 때(49.3%)보다 전체 지식 문장을 제공했을 때(54.7%) 성능이 더 높았으며, 이는 상세한 텍스트 정보가 추론에 결정적임을 시사한다.
- **상한선 확인 (Oracle):** 정답 엔티티를 직접 제공하는 Oracle 설정에서도 모델들이 완벽한 성능을 내지 못했다. 이는 외부 지식이 제공되더라도 LALM이 이를 정확히 활용하여 추론하는 능력 자체가 아직 부족함을 의미한다.

## 🧠 Insights & Discussion

본 연구는 LALM이 대규모 사전 학습을 거쳤음에도 불구하고, 특정 명명 엔티티에 대한 세부적인 세계 지식을 내재적으로 보유하고 있지 않음을 정량적으로 보여주었다. 제안된 AEL + $KA^2LM$ 파이프라인은 모델의 구조를 변경하지 않고도 성능을 높일 수 있는 유연한 해결책을 제시한다.

**한계점 및 논의사항:**

- **단일 엔티티 가정:** 각 오디오에 단 하나의 명명 엔티티만 존재한다고 가정하여 구축된 데이터셋이므로, 여러 엔티티가 복합적으로 등장하는 실제 상황에서의 일반화 성능은 검증되지 않았다.
- **언어 제한:** 영어 데이터셋만을 사용했으므로 다국어 환경에서의 적용 가능성에 대한 추가 연구가 필요하다.
- **합성 데이터의 노이즈:** 데이터셋의 약 5% 미만이 합성 과정에서 노이즈를 포함하고 있어, 실제 데이터와의 괴리가 있을 수 있다.
- **비판적 시각:** 외부 지식을 텍스트로 직접 주입하는 방식은 추론 속도를 늦출 수 있으며, 모델이 지식 베이스의 텍스트에 과도하게 의존하여 오디오 자체의 맥락을 무시할 위험이 있다.

## 📌 TL;DR

본 논문은 오디오 이해와 외부 세계 지식을 결합해야 하는 새로운 벤치마크 **Audiopedia (s, m, r-AQA)**를 제안하고, 이를 해결하기 위해 오디오 내 엔티티를 지식 베이스에 연결하여 프롬프트를 증강하는 **AEL + $KA^2LM$** 프레임워크를 제시하였다. 실험 결과, 기존 LALM들은 지식 집약적 태스크에 매우 취약하며, 제안된 외부 지식 주입 방식이 성능 향상에 필수적임을 확인하였다. 이 연구는 향후 오디오 언어 모델이 단순한 인식을 넘어 고차원적인 지식 추론 능력을 갖추게 하는 방향성을 제시한다.
