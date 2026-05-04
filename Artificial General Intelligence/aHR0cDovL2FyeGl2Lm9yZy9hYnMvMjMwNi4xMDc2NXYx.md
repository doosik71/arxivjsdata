# Path to Medical AGI: Unify Domain-specific Medical LLMs with the Lowest Cost

Juexiao Zhou, Xiuying Chen, Xin Gao (2023)

## 🧩 Problem to Solve

본 논문은 의료 분야의 인공 일반 지능(Medical Artificial General Intelligence, Medical AGI)을 구현하기 위해, 서로 다른 의료 도메인에 특화된 대규모 언어 모델(LLM)들을 효율적으로 통합하는 문제를 해결하고자 한다.

의료 분야에서 범용적인 모델을 구축하는 것은 매우 중요하다. 그러나 다음과 같은 현실적인 제약이 존재한다:

1. **데이터 수집의 어려움**: 의료 데이터는 개인정보 보호 제한이 매우 엄격하며, 공개적으로 사용 가능한 데이터셋이 부족하여 다양한 도메인을 아우르는 통합 데이터셋을 구축하기 어렵다.
2. **리소스 낭비**: 현재는 피부과, 엑스레이, 병리 분석 등 각 도메인별로 특화된 멀티모달 LLM들이 개별적으로 개발되고 있다. 사용자가 자신의 질문에 맞는 모델을 직접 찾아야 하며, 각 모델마다 동일한 Image Encoder나 LLM backbone을 중복해서 저장하고 로드해야 하므로 저장 공간과 계산 리소스 낭비가 심하다.
3. **모델 통합의 비현실성**: 모든 의료 데이터를 수집하여 하나의 거대 모델을 다시 학습시키는 것은 데이터 공유의 한계로 인해 불가능에 가깝다.

따라서 본 논문의 목표는 추가적인 재학습 없이도 사용자의 질문을 분석하여 최적의 도메인 전문가 모델을 자동으로 선택하고 연결하는, 저비용 고효율의 통합 프레임워크인 MedAGI를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **'Adaptive Expert Selection'** 알고리즘을 통해, 공유된 인프라(Vision Encoder 및 LLM) 위에서 도메인별로 최적화된 **'Alignment Layer(정렬 층)'**만을 동적으로 교체하여 사용하는 것이다.

중심적인 설계 직관은 다음과 같다:

- **모듈형 구조**: 모든 의료 모델이 유사한 구조(Vision Encoder $\rightarrow$ Alignment Layer $\rightarrow$ LLM)를 가진다는 점에 착안하여, 변하지 않는 부분(Encoder, LLM)은 공유하고, 도메인 지식이 응축된 Alignment Layer만을 데이터베이스화하여 관리한다.
- **자동화된 전문가 선택**: 사용자의 질문 텍스트를 분석하여 어떤 의료 도메인에 해당하는지를 판별하고, 해당 도메인의 Alignment Layer를 즉시 적용함으로써 사용자가 도메인을 직접 선택해야 하는 번거로움을 제거한다.
- **확장성(Future-proof)**: 새로운 도메인의 모델이 개발되더라도, 전체 시스템을 재학습시킬 필요 없이 해당 모델의 Alignment Layer만 데이터베이스에 추가하면 즉시 시스템에 통합될 수 있다.

## 📎 Related Works

논문에서는 기존의 의료 멀티모달 LLM 접근 방식을 두 가지로 분류하여 설명한다.

1. **End-to-End 학습 방식**: Vision Encoder와 LLM을 결합하여 전체를 학습시키는 방식(예: LLaVA-Med, PathAsst)이다. 이 방식은 강력하지만, 앞서 언급한 대로 방대한 양의 다중 도메인 데이터를 수집해야 한다는 치명적인 한계가 있다.
2. **Alignment Layer 기반 방식**: 사전 학습된 Image Encoder와 LLM 사이에 작은 정렬 층(Alignment Layer)을 두고, 도메인 특화 데이터로 이 부분만 미세 조정(Fine-tuning)하는 방식(예: SkinGPT-4, XrayGPT, XrayChat)이다. 이 방식은 적은 데이터와 적은 파라미터 업데이트만으로도 가능하여 더 현실적이다.

MedAGI는 후자의 방식을 채택하되, 개별적으로 존재하는 이러한 Alignment Layer들을 하나의 플랫폼으로 통합하여 **'도메인에 구애받지 않는(Domain-agnostic)'** 인터페이스를 제공한다는 점에서 기존 연구들과 차별화된다.

## 🛠️ Methodology

### 전체 시스템 구조

MedAGI의 파이프라인은 **[이미지 입력 $\rightarrow$ 시각적 특징 추출 $\rightarrow$ 전문가 레이어 선택 $\rightarrow$ 텍스트 생성]**의 순서로 진행된다.

1. **Vision Encoding**: 입력된 의료 이미지는 Vision Transformer (ViT)와 Q-Former를 거쳐 이미지 임베딩(Image Embedding)으로 변환된다.
2. **Adaptive Model Selection**: 사용자의 질문을 분석하여 데이터베이스에 저장된 여러 전문가 모델의 설명(Description)과 비교하고, 가장 적합한 전문가 Alignment Layer를 선택한다.
3. **Visual-Language Alignment**: 선택된 Alignment Layer가 Q-Former에서 나온 시각적 표현을 LLM이 이해할 수 있는 형태로 정렬한다.
4. **Response Generation**: 최종적으로 LLM(Alpaca 또는 Vicuna 등)이 정렬된 정보와 질문을 바탕으로 진단 결과나 설명을 텍스트로 생성한다.

### 전문가 선택 알고리즘 (Adaptive Selection Algorithm)

사용자의 질문 $q$와 각 모델의 설명 $d_j$ 사이의 유사도를 계산하여 최적의 모델을 선택한다.

- **인코딩**: BERT 모델을 사용하여 사용자 질문 $q = \{w_{q1}, \dots, w_{qL_q}\}$와 $j$번째 모델 설명 $d_j = \{w_{d,j1}, \dots, w_{d,jL_d}\}$를 벡터로 변환한다.
  $$ \langle h_{q1}, \dots, h_{qL_q} \rangle = \text{Enc}(w_{q1}, \dots, w_{qL_q}) $$
  $$ \langle h_{d,j1}, \dots, h_{d,jL_d} \rangle = \text{Enc}(w_{d,j1}, \dots, w_{d,jL_d}) $$
- **평균 풀링(Mean-pooling)**: 토큰 단위의 벡터들을 하나의 문장 임베딩 벡터 $u$ (사용자 질문)와 $v_j$ (모델 설명)로 압축한다.
  $$ u = \text{Mean-pooling}(\langle h_{q1}, \dots, h_{qL_q} \rangle) $$
- **유사도 계산**: 코사인 유사도(Cosine Similarity)를 통해 질문과 모델 설명 간의 유사도 점수 $s_j$를 계산한다.
  $$ s_j = \text{similarity}(u, v_j) $$
- **최종 선택**: 가장 높은 $s_j$ 값을 가진 모델의 Alignment Layer를 선택하여 추론에 사용한다.

### 학습 절차

본 연구에서는 MiniGPT-4를 백본으로 하여 SkinGPT-4, XrayChat, PathologyChat의 Alignment Layer를 구현하였다.

- **학습 설정**: 최대 5 epoch, epoch당 5000 iteration, 학습률 $1e-4$, 배치 사이즈 2를 적용하였다.
- **학습 자원**: NVIDIA V100 (32GB) GPU 2장을 사용하였으며, 전체 미세 조정에 약 9시간이 소요되었다.

## 📊 Results

### 실험 설정

- **평가 도메인**: 피부과(Dermatology), 엑스레이(X-ray), 병리 분석(Pathology)의 3가지 영역.
- **비교 대상**: MedAGI, SkinGPT-4, XrayChat, PathologyChat, MiniGPT-4.
- **데이터셋**: SKIN-CON, Dermnet (피부), Open-i, MIMIC CXR (엑스레이), H&E-stained gastric slides (병리).

### 정성적 결과 분석

논문은 수치적인 정량 지표 대신, 실제 사례(Case)에 대한 모델별 답변을 비교하는 정성적 평가 결과를 제시한다.

1. **도메인 내 성능**: 각 도메인 특화 모델(SkinGPT-4, XrayChat, PathologyChat)은 자신의 전문 분야에서 정확한 답변을 내놓았다. MedAGI는 적절한 전문가 레이어를 선택함으로써 이들 특화 모델과 동일한 수준의 정답을 생성하였다.
2. **교차 도메인 성능**: 특화 모델들은 자신의 분야가 아닌 이미지가 입력되었을 때, "이미지를 볼 수 없다"고 하거나 완전히 잘못된 진단(예: 피부 이미지에 대해 엑스레이 진단을 내림)을 내리는 한계를 보였다. 반면, MedAGI는 질문을 분석해 적절한 레이어를 선택함으로써 어떤 도메인의 이미지라도 유연하게 대응하였다.
3. **범용 모델(MiniGPT-4)과의 비교**: 일반적인 멀티모달 모델인 MiniGPT-4는 의료 전문 지식이 부족하여 구체적인 진단보다는 일반적인 묘사에 그치거나 부정확한 답변을 생성하는 경향을 보였다.

## 🧠 Insights & Discussion

### 강점 및 가치

MedAGI는 **'최소 비용으로 전문가 모델들을 통합'**했다는 점에 큰 의의가 있다. 전체 모델을 다시 학습시키지 않고 작은 Alignment Layer만 교체하는 방식이기에, 컴퓨팅 자원이 제한된 환경에서도 Medical AGI로 나아갈 수 있는 현실적인 경로를 제시하였다. 또한, 새로운 전문 모델이 등장할 때마다 단순 추가만으로 성능을 확장할 수 있는 뛰어난 확장성을 가진다.

### 한계 및 미해결 과제

1. **정량적 평가 부족**: 논문에서 제시한 결과가 주로 몇 가지 사례에 대한 정성적 비교에 치중되어 있다. 대규모 벤치마크 데이터셋을 통한 정량적 성능 지표(예: Accuracy, F1-score 등)가 명시되지 않아 객관적인 성능 향상 폭을 확인하기 어렵다.
2. **BERT 기반 선택의 의존성**: 전문가 선택 과정이 BERT와 코사인 유사도에 의존하고 있다. 만약 사용자의 질문이 모호하거나 여러 도메인이 섞여 있는 경우, 잘못된 전문가 레이어를 선택할 가능성이 있으며 이에 대한 예외 처리 메커니즘이 부족하다.
3. **추론 지연 시간**: 이미지를 처리하고 BERT로 모델을 선택한 뒤 다시 LLM을 통해 생성하는 과정에서 발생하는 추가적인 latency에 대한 분석이 없다.

## 📌 TL;DR

본 논문은 의료 데이터의 희소성과 개인정보 보호 문제로 인해 통합 모델 구축이 어려운 상황에서, **도메인별 전문 LLM들의 Alignment Layer만을 동적으로 선택하여 사용하는 MedAGI 프레임워크**를 제안한다. BERT 기반의 적응형 선택 알고리즘을 통해 사용자의 질문에 맞는 전문가 모델을 자동으로 연결함으로써, 추가 학습 없이도 다양한 의료 도메인을 통합 지원하는 저비용/고확장성 솔루션을 구현하였다. 이는 향후 다양한 의료 전문 모델들이 출시됨에 따라 이를 효율적으로 통합 관리하는 표준 플랫폼으로 활용될 가능성이 높다.
