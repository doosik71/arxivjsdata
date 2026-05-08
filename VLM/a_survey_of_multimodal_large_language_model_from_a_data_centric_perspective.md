# A Survey of Multimodal Large Language Model from A Data-centric Perspective

Tianyi Bai, Hao Liang, Binwang Wan, Yanran Xu, Xi Li, Shiyu Li, Ling Yang, Bozhou Li, Yifan Wang, Bin Cui, Ping Huang, Jiulong Shan, Conghui He, Binhang Yuan, Wentao Zhang (2024)

## 🧩 Problem to Solve

최근 Multimodal Large Language Models (MLLMs)는 텍스트, 이미지, 비디오, 오디오 등 다양한 모달리티를 통합하여 뛰어난 이해 및 생성 능력을 보여주고 있다. 하지만 지금까지의 대부분의 MLLM 연구는 모델의 아키텍처를 수정하여 모달리티 정보를 어떻게 효율적으로 활용할 것인가라는 '모델 중심(model-centric)' 관점에 치중되어 있었다.

본 논문은 모델의 성능이 아키텍처뿐만 아니라 학습 데이터의 규모와 품질에 의해 결정된다는 점에 주목한다. 특히 Scaling Law에 따라 데이터의 양이 중요하며, 정교하게 큐레이션된 데이터셋은 작은 모델로도 큰 모델에 필적하는 성능을 낼 수 있다는 점이 기존 연구를 통해 밝혀졌다. 그럼에도 불구하고 MLLM을 위한 데이터 큐레이션과 활용 방안에 대한 종합적인 연구는 부족한 실정이다. 따라서 본 논문의 목표는 MLLM의 데이터 수집, 처리, 선택, 그리고 평가에 이르는 전 과정을 '데이터 중심(data-centric)' 관점에서 분석하고 체계적으로 정리하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 MLLM의 발전 과정을 데이터 관점에서 재구성하여 다음과 같은 종합적인 프레임워크를 제시한 것에 있다.

1. **데이터 중심의 새로운 관점 제공**: 텍스트, 이미지, 비디오, 오디오를 포함한 다양한 모달리티를 아우르는 데이터 중심의 MLLM 리뷰를 제공한다.
2. **데이터 준비 및 관리 파이프라인 체계화**: MLLM의 학습 단계인 Pre-training(사전 학습)과 Adaptation(적응/미세 조정) 단계에서 데이터가 어떻게 수집, 처리, 선택되는지에 대한 상세 파이프라인을 정의한다.
3. **데이터 평가 방법론 및 벤치마크 분석**: 데이터셋 자체의 품질을 평가하는 지표와 MLLM의 성능을 측정하는 데이터 중심의 평가 벤치마크를 분석한다.
4. **향후 연구 방향 제시**: 데이터 처리 시스템의 부재, 데이터 양에 따른 Emergent Abilities 분석, 프록시 모델(Proxy Model)을 이용한 최적화 등 미래 연구 과제를 제안한다.

## 📎 Related Works

기존의 LLM 및 MLLM 관련 연구들은 주로 모델의 구조적 개선(Architectural enhancements)에 집중한 model-centric 접근 방식을 취해왔다. 또한, 최근 LLM을 위한 데이터 관리 및 선택 방법론에 대한 연구들이 등장하였으나, 이는 주로 텍스트 전용 모델에 국한되어 있었다.

Data-centric AI (DCAI)라는 일반적인 관점의 연구는 존재하지만, 이를 MLLM이라는 특수한 도메인에 적용하여 데이터 파이프라인 전체를 분석한 사례는 드물었다. 본 논문은 이러한 공백을 메우기 위해, 단순한 모델 구조의 변경이 아닌 학습 코퍼스 데이터셋이 모델 성능에 미치는 포괄적인 영향을 분석한다는 점에서 기존 연구들과 차별화된다.

## 🛠️ Methodology

본 논문은 MLLM의 데이터 파이프라인을 크게 세 단계(Pre-processing $\rightarrow$ Pre-training $\rightarrow$ Adaptation)로 구분하여 설명한다.

### 1. 데이터 수집 및 처리 (Data Collecting and Processing)

- **수집원 (Sources)**: 일반 웹페이지(CommonCrawl), 소셜 미디어(Reddit, X, YouTube), 학술 논문(arXiv), 도서, 코드 저장소, 도메인 특화 소스(의료, 법률, 금융) 등에서 데이터를 수집한다.
- **필터링 (Filtering)**:
  - **텍스트**: 언어 식별(Langdetect, FastText) 및 유해 콘텐츠/불필요한 태그 제거.
  - **이미지**: 저해상도 제거, 부적절한 종횡비 제거, NSFW 콘텐츠 필터링.
  - **비디오**: 장면 전환 검출, 모션 분석(Optical Flow), OCR을 통한 텍스트 과다 포함 클립 제거.
- **중복 제거 (Deduplication)**:
  - **Exact**: 정확한 문자열 일치 제거.
  - **Approximate**: MinHash, SimHash 등을 이용한 근사적 중복 제거.
  - **Semantic**: Sentence-BERT 등을 이용해 임베딩 공간에서 의미적 중복 제거.
- **데이터 강화 (Enhancement)**: 캡션 재작성(Caption rewriting)을 통한 텍스트 품질 향상 및 입력 해상도(Resolution) 상향 조정을 통한 세부 특징 포착 능력 강화.

### 2. 데이터 중심의 사전 학습 (Data-Centric Pre-training)

사전 학습은 크게 두 단계로 진행된다. 첫째는 LLM 백본과 각 모달리티 인코더를 개별적으로 학습시키는 단계이며, 둘째는 Projector를 통해 인코더의 특징을 LLM의 임베딩 공간으로 매핑하며 통합 학습시키는 단계이다.

- **Domain Mixture**: 위키피디아, 도서 등 서로 다른 도메인의 데이터 비율을 최적화하는 문제로, 최근에는 작은 프록시 모델을 사용하여 최적의 가중치를 예측하는 방법(DoReMi 등)이 사용된다.
- **Modality Mixture**: 이미지-캡션, 인터리브(Interleaved) 문서, 텍스트 전용 데이터의 혼합 비율을 결정한다. (예: MM1 모델의 경우 5:5:1 비율이 최적임을 발견)
- **품질 선택 (Quality Selection)**: 모든 데이터를 사용하는 대신 CLIP score나 분포 기반의 선택 방법을 통해 고품질 데이터만 선별하여 학습 효율을 높인다.

### 3. 데이터 중심의 적응 (Data-Centric Adaptation)

사전 학습된 모델을 특정 작업이나 인간의 선호도에 맞게 조정하는 단계이다.

- **Supervised Fine-Tuning (SFT)**:
  - **데이터 생성**: 단순 캡셔닝, VQA(Visual Question Answering), 추론(Reasoning), 분류 작업 등으로 구분하여 Instruction-Response 쌍을 생성한다. GPT-4와 같은 강력한 모델을 사용하여 기존 캡션을 고품질의 지시어 형태로 변환하는 기법이 주로 사용된다.
  - **데이터 선택**: Coreset 기반(기하학적 대표성), LLM 기반(품질 스코어링), Gradient 기반(영향력 분석) 방법론을 통해 핵심 데이터셋을 선별한다.
- **인간 선호도 정렬 (Human Preference Alignment)**: RLHF(Reinforcement Learning from Human Feedback)를 통해 모델의 응답을 인간의 가치(Helpfulness, Honesty, Harmlessness)에 맞게 조정한다.

## 📊 Results

본 논문은 서베이 논문으로서 특정 단일 실험 결과보다는 기존 연구들의 데이터 관련 정량적/정성적 결과를 종합하여 제시한다.

- **데이터 비율의 영향**: MM1 연구를 통해 이미지-캡션, 인터리브 데이터, 텍스트 데이터의 최적 혼합 비율이 존재하며, 이를 통해 Zero-shot 및 Few-shot 성능이 향상됨을 확인하였다.
- **해상도의 중요성**: 입력 해상도를 224px에서 336px, 448px, 나아가 896px(Monkey 모델)까지 높였을 때, 세밀한 객체 인식 및 세그멘테이션 성능이 비약적으로 상승함을 보여준다.
- **SFT 데이터 효율성**: LIMA 연구 등을 인용하여, 방대한 양의 데이터보다 매우 정교하게 큐레이션된 소량의 고품질 지시어 데이터셋이 모델의 Instruction-following 능력을 부여하는 데 더 효과적임을 강조한다.
- **평가 지표**: 데이터 다양성 측정에는 Vendi Score가, 텍스트-이미지 일치도(Faithfulness) 측정에는 Faith Score와 CHAIR(Hallucination 측정) 등이 유용하게 사용되고 있음을 정리하였다.

## 🧠 Insights & Discussion

**강점 및 분석**:
본 논문은 MLLM의 성능 향상을 위해 모델 구조의 변경이라는 고전적인 방식에서 벗어나, 데이터의 '생애 주기(수집 $\rightarrow$ 처리 $\rightarrow$ 학습 $\rightarrow$ 적응 $\rightarrow$ 평가)' 전체를 조망했다는 점에서 매우 높은 학술적 가치를 지닌다. 특히 모달리티별(이미지, 비디오, 오디오, 3D) 데이터 처리 특성을 세분화하여 분석한 점이 돋보인다.

**한계 및 논의사항**:

1. **데이터 처리 시스템의 부재**: LLM을 위한 Data-Juicer 같은 시스템은 존재하지만, 비디오나 오디오 등 복합 모달리티를 통합적으로 처리할 수 있는 MLLM 전용 데이터 파이프라인 시스템은 여전히 부족하다.
2. **Scaling Law의 불명확성**: 텍스트 전용 모델에서는 데이터 양과 성능의 관계가 명확히 규명되었으나, MLLM에서 데이터의 양이 Emergent Abilities(창발적 능력)에 구체적으로 어떤 영향을 미치는지에 대한 분석이 더 필요하다.
3. **망각 문제 (Catastrophic Forgetting)**: 순차적인 학습 단계(Pre-training $\rightarrow$ SFT $\rightarrow$ RLHF)를 거치면서 기초적인 언어 능력을 잃지 않고 다중 모달리티 능력을 습득하게 하는 Lifelong Learning 전략이 필수적이다.

## 📌 TL;DR

본 논문은 MLLM의 성능 결정 요인으로 '데이터'에 집중하여, 데이터 수집부터 전처리, 사전 학습 시의 모달리티 혼합, SFT를 위한 고품질 지시어 생성 및 선택, 그리고 데이터 중심의 평가 방법론까지의 전체 파이프라인을 체계적으로 정리한 종합 서베이 보고서이다. 이 연구는 향후 MLLM 연구자들이 모델 구조 변경에 매몰되지 않고, 데이터 큐레이션을 통해 효율적으로 모델 성능을 끌어올릴 수 있는 가이드라인을 제공하며, 특히 데이터 효율적 학습과 다중 모달리티 정렬 연구에 중요한 기초 자료가 될 것으로 보인다.
