# Transformers in Healthcare: A Survey

Subhash Nerella, Sabyasachi Bandyopadhyay, Jiaqing Zhang, Miguel Contreras, Scott Siegel, Aysegul Bumin, Brandon Silva, Jessica Sena, Benjamin Shickel, Azra Bihorac, Kia Khezeli, Parisa Rashidi (2023)

## 🧩 Problem to Solve

본 논문은 의료 산업에서 폭발적으로 증가하고 있는 데이터(전 세계 데이터 생태계의 약 30% 차지)를 효율적으로 분석하고 활용하기 위한 방안으로, 최근 딥러닝 분야의 핵심 아키텍처인 Transformer의 도입 현황과 그 영향을 분석하고자 한다. 

의료 데이터는 의료 영상, 구조화 및 비구조화된 전자 건강 기록(Electronic Health Records, EHR), 소셜 미디어 텍스트, 생체 신호, 생체 분자 서열 등 매우 다양한 모달리티(Modality)로 구성되어 있다. 기존의 순환 신경망(RNN)은 순차적 처리 방식으로 인해 학습 속도가 느리고, 합성곱 신경망(CNN)은 수용 영역(Receptive Field)의 제한으로 인해 전역적 문맥(Global Context)을 파악하는 데 한계가 있었다. 따라서 본 논문은 Transformer 아키텍처가 이러한 다양한 의료 데이터 분석에 어떻게 적용되었는지 종합적으로 검토하고, 임상 진단, 보고서 생성, 데이터 재구성, 약물/단백질 합성 등의 태스크에서 거둔 성과를 체계적으로 정리하는 것을 목표로 한다.

## ✨ Key Contributions

본 논문의 핵심 기여는 Transformer 모델이 의료 분야의 다중 모달리티 데이터에 적용된 사례를 PRISMA(Preferred Reporting Items for Systematic Reviews and Meta-Analyses) 가이드라인에 따라 체계적으로 분석한 종합 서베이를 제공했다는 점이다.

중심적인 설계 아이디어는 Transformer의 **Attention 메커니즘**이 데이터 내의 장거리 의존성(Long-range dependencies)을 효과적으로 캡처할 수 있다는 점을 활용하여, 이를 의료 도메인의 특수한 데이터 구조(예: EHR의 시계열 구조, 의료 영상의 공간적 구조, 단백질의 서열 구조)에 최적화하는 것이다. 또한, 단순히 모델의 나열에 그치지 않고 계산 비용, 모델의 해석 가능성(Interpretability), 공정성(Fairness), 윤리적 영향 및 환경적 영향(탄소 배출)과 같은 실무적/윤리적 쟁점을 심도 있게 논의하였다.

## 📎 Related Works

논문은 Transformer 이전의 주류 모델이었던 RNN과 그 변형 모델들의 한계를 지적한다. RNN은 데이터를 순차적으로 처리해야 하므로 병렬화가 불가능하여 학습 시간이 매우 길다는 단점이 있다. 반면, Transformer는 Scaled Dot-Product Attention을 통해 병렬 처리를 가능하게 하여 대규모 사전 학습(Pre-training)을 가능케 하였다.

또한, 기존의 서베이 논문들이 주로 의료 영상(Medical Imaging)이나 생의학 언어 모델(Biomedical Language Models)과 같이 특정 모달리티에 집중했던 것과 달리, 본 연구는 NLP, EHR, 소셜 미디어, 의료 영상, 생체 신호, 생체 분자 서열을 모두 아우르는 통합적인 관점을 제시함으로써 기존 연구와의 차별성을 갖는다.

## 🛠️ Methodology

### 1. Transformer 기본 아키텍처
Transformer는 인코더-디코더 구조의 적층형 신경망으로, 핵심은 **Attention 메커니즘**이다. 입력 임베딩은 Query($Q$), Key($K$), Value($V$) 세 가지 역할로 변환된다.

- **Scaled Dot-Product Attention**: $Q$와 $K$의 유사도를 계산하여 $V$에 가중치를 부여하는 방식이다. 수식은 다음과 같다.
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
여기서 $\sqrt{d_k}$는 그래디언트 소실 문제를 완화하기 위한 스케일링 인자이다.

- **Multi-Head Attention**: 단일 Attention 대신 여러 개의 Head를 병렬로 배치하여, 데이터의 다양한 상관관계를 동시에 학습한다. 각 Head의 출력은 연결(Concatenate)된 후 선형 변환된다.

- **Position-wise Feed-Forward Network (FFN)**: Attention 모듈의 출력은 두 개의 선형 층과 ReLU 활성화 함수로 구성된 FFN을 통과하며, 각 위치에서 독립적으로 연산된다.
$$\mathcal{F}(\mathcal{H}) = \text{ReLU}(\mathcal{H}W_1 + b_1)W_2 + b_2$$

- **Positional Encoding (PE)**: Transformer는 병렬 처리를 하므로 순서 정보가 없다. 이를 해결하기 위해 사인(sine)과 코사인(cosine) 함수를 이용한 PE 벡터를 입력 임베딩에 더해 위치 정보를 주입한다.

### 2. 주요 변형 아키텍처
- **BERT (Encoder-only)**: 양방향 문맥을 학습하기 위해 Masked Language Modeling(MLM)과 Next Sentence Prediction(NSP)을 사용한다. 의료 분야에서는 BioBERT, ClinicalBERT 등으로 확장되어 사용된다.
- **ViT (Vision Transformer, Encoder-only)**: 이미지를 고정된 크기의 패치(Patch)로 나누어 시퀀스로 변환한 후 Transformer 인코더에 입력하는 구조이다.
- **LLMs (Decoder-only)**: GPT-4, PaLM과 같이 수십억 개의 파라미터를 가진 거대 언어 모델로, 주로 자기회귀(Autoregressive) 방식으로 다음 토큰을 예측하며 생성 태스크에 특화되어 있다.

### 3. 의료 분야 적용 파이프라인
- **Clinical NLP**: 일반 BERT를 의료 말뭉치(PubMed, MIMIC-III 등)로 추가 학습시킨 BioBERT, ClinicalBERT 등을 활용하여 NER, QA, 관계 추출(RE) 등을 수행한다.
- **Medical Imaging**: CNN의 로컬 특징 추출 능력과 Transformer의 글로벌 문맥 파악 능력을 결합한 Hybrid 구조(예: TransUNet)가 주로 사용된다.
- **Structured EHR**: ICD 코드를 토큰으로 처리하여 BERT 구조에 입력하는 BEHRT와 같은 모델을 통해 미래 질병을 예측한다.

## 📊 Results

본 논문은 특정 실험 결과보다는 다양한 연구 사례의 성과를 정리하여 제시한다.

- **데이터셋 및 지표**: MIMIC-III, PubMed, Synapse, ACDC, BCV 등 공공 의료 데이터셋이 널리 사용되었으며, 성능 지표로는 F1-score, Pearson correlation, Dice coefficient(분할 작업) 등이 사용되었다.
- **주요 정량적 결과**:
    - **Clinical NLP**: BioBERT와 ClinicalBERT가 일반 BERT보다 의료 도메인 텍스트 마이닝 작업에서 우수한 성능을 보였다.
    - **의료 영상 분할**: TransUNet과 Swin-Unet 등의 모델이 기존 CNN 기반의 U-Net보다 전역적 문맥 파악 능력이 뛰어나 다기관 분할(Multi-organ segmentation) 등에서 더 높은 정확도를 기록하였다.
    - **약물 합성**: Molecular Transformer가 약물 대사 예측 및 단백질-약물 상호작용 예측에서 기존 SOTA 모델들을 상회하는 결과를 보였다.
    - **LLM 성능**: Med-PaLM 2와 같은 모델이 의료 면허 시험(USMLE)에서 전문가 수준의 점수를 획득하며 임상 지식 학습 능력을 입증하였다.

## 🧠 Insights & Discussion

### 강점 및 가능성
Transformer는 의료 데이터의 본질적인 특성인 '복잡한 상호작용 네트워크'와 '장거리 의존성'을 모델링하는 데 매우 적합하다. 특히 Multi-modal Fusion(예: EHR과 의료 영상의 결합)을 통해 더 정확한 환자 표현(Patient Representation)을 생성할 수 있다는 점이 큰 강점이다.

### 한계 및 비판적 해석
- **계산 복잡도**: Attention 연산의 시간 복잡도가 $O(n^2 \cdot d)$이므로, 매우 긴 시퀀스(예: 고해상도 이미지나 긴 의료 기록)를 처리할 때 메모리와 계산 비용이 기하급수적으로 증가한다.
- **해석 가능성(Interpretability)**: Attention weight 시각화가 어느 정도 단서를 제공하지만, 이것이 반드시 모델의 실제 의사결정 근거와 일치하지 않는다는 점이 지적된다. 의료 분야에서는 '왜' 그런 진단이 나왔는지가 매우 중요하므로, 이는 여전히 해결해야 할 과제이다.
- **환경 및 비용 문제**: 거대 모델(LLM)의 학습 과정에서 발생하는 막대한 탄소 배출과 수백만 달러에 달하는 학습 비용은 의료 현장의 실무적 도입에 큰 장벽이 된다.
- **데이터 프라이버시 및 편향**: HIPAA와 같은 엄격한 규제로 인해 양질의 데이터 확보가 어렵고, 특정 인구 집단에 편향된 데이터로 학습된 모델은 의료 불평등을 초래할 위험이 있다.

### 해결 방향
논문은 모델 압축(Pruning, Quantization, Knowledge Distillation)을 통한 경량화, 데이터 프라이버시 보호를 위한 **연합 학습(Federated Learning)**의 도입, 그리고 인간의 가치와 정렬된 AI Alignment 연구의 필요성을 강조한다.

## 📌 TL;DR

본 논문은 Transformer 아키텍처가 의료 영상, EHR, 생체 신호, 분자 서열 등 다양한 의료 데이터 모달리티에 어떻게 적용되었는지를 체계적으로 분석한 서베이 보고서이다. Transformer는 장거리 의존성 모델링 능력을 통해 의료 진단 및 약물 설계 등의 분야에서 획기적인 성능 향상을 가져왔으나, 높은 계산 비용, 낮은 해석 가능성, 데이터 프라이버시 및 윤리적 편향성이라는 중대한 과제를 안고 있다. 향후 연구는 모델의 효율적 경량화와 연합 학습을 통한 프라이버시 보존, 그리고 임상 현장에서 신뢰할 수 있는 해석 가능한 AI를 구축하는 방향으로 나아가야 할 것이다.