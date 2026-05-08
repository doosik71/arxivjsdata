# Knowledge-Guided Prompt Learning for Lifespan Brain MR Image Segmentation

Lin Teng, Zihao Zhao, Jiawei Huang, Zehong Cao, Runqi Meng, Feng Shi, and Dinggang Shen (2024)

## 🧩 Problem to Solve

본 논문은 인간의 전 생애 주기(lifespan)에 걸친 뇌 MRI 영상의 조직(tissue) 및 구조(structure)를 자동으로 정밀하게 분할(segmentation)하는 문제를 해결하고자 한다. 뇌 MRI 분할은 뇌 발달 분석과 신경 퇴행성 질환 진단에 필수적이지만, 다음과 같은 주요 난관이 존재한다.

첫째, 생애 주기에 따른 뇌의 형태학적 변화가 매우 심하다. 초기 뇌 발달 단계의 급격한 변화, 노화로 인한 뇌 조직의 감소, 그리고 알츠하이머병(AD)과 같은 질환으로 인한 변형 등으로 인해 뇌의 외형적 변동성이 크다.

둘째, 정교하게 라벨링된 데이터셋의 부족이다. 수동 주석(manual annotation) 작업은 시간이 많이 소요되고 전문가의 숙련도가 요구되어 대규모의 고품질 데이터를 확보하기 어렵다.

따라서 본 연구의 목표는 텍스트 기반의 생물학적 지식을 활용하여 다양한 연령대의 뇌 구조적 변동성을 효과적으로 학습하고, 적은 양의 데이터로도 강건한 분할 성능을 내는 Knowledge-Guided Prompt Learning (KGPL) 프레임워크를 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 시각적 프롬프트 튜닝(Visual Prompt Tuning, VPT)의 개념을 확장하여, 무작위(random) 임베딩 대신 **생물학적 지식이 반영된 임베딩(knowledge-driven embeddings)**을 모델의 입력 공간에 주입하는 것이다.

단순히 이미지 정보만 사용하는 것이 아니라, 환자의 연령, 성별, 질병 상태와 같은 텍스트 기반의 생물학적 속성 정보를 BiomedCLIP의 텍스트 인코더를 통해 임베딩으로 변환하고, 이를 프롬프트로 활용함으로써 모델이 연령대별 뇌 구조의 특성을 더 잘 이해하도록 유도한다. 이는 모델이 이미지의 픽셀 정보뿐만 아니라 해부학적 변동성과 생물학적 과정 사이의 세만틱 관계를 캡처할 수 있게 한다.

## 📎 Related Works

기존의 뇌 MRI 분할은 FSL이나 FreeSurfer와 같은 소프트웨어 패키지에 의존해 왔으나, 처리 시간이 매우 길고 전문가의 수동 수정이 필요하다는 한계가 있다. 최근에는 U-Net, UNETR, Swin UNETR와 같은 딥러닝 기반 모델들이 등장하며 성능을 높였으나, 대부분 특정 시점의 데이터에 집중하며 텍스트 형태의 보완적 정보(병리 및 해부학적 맥락)를 간과하여 다양한 연령대에 대한 일반화 성능이 떨어진다.

또한, 데이터 부족 문제를 해결하기 위해 전이 학습(transfer learning)이나 파인튜닝(fine-tuning) 전략이 사용되었다. 특히 VPT와 같은 프롬프트 튜닝 방식이 제안되었으나, 이는 무작위로 초기화된 파라미터를 사용하므로 모델 학습 초기 단계에서 방향성을 잡는 데 어려움이 있을 수 있다. 본 논문은 이러한 무작위성 대신 BiomedCLIP을 통한 '지식 가이드'를 제공함으로써 기존 방식과 차별화된다.

## 🛠️ Methodology

본 연구는 크게 두 단계의 프레임워크로 구성된다.

### 1. Vision Pre-training

먼저 서브-옵티멀(sub-optimal) 라벨(예: FreeSurfer로 생성된 자동 라벨)을 가진 대규모 데이터셋을 사용하여 모델을 사전 학습시킨다. 이 과정은 두 단계로 나뉜다.

- **조직 분할(Tissue Segmentation):** T1-weighted MR 영상을 입력받아 뇌 조직 마스크를 예측하며, 손실 함수로 Dice Loss를 사용한다.
$$ \text{Dice Loss} = 1 - \frac{2 \times |P \cap G|}{|P| + |G|} $$
여기서 $P$는 예측 결과, $G$는 Ground Truth를 의미한다.
- **구조 분할(Structure Parcellation):** 예측된 조직 분할 결과를 다시 입력으로 사용하여 세부 뇌 구조를 분할하며, Dice Loss와 Focal Loss의 조합을 사용한다.
$$ \text{Loss} = \text{Dice Loss} - \alpha \times (1 - p_t)^\gamma \times \log(p_t) $$
($\alpha=100, \gamma=0.2$)

### 2. Fine-tuning with Knowledge-wise Prompt (KGPL)

사전 학습된 모델을 소규모의 고품질 수동 라벨 데이터셋으로 정밀 튜닝하는 단계이다.

**Knowledge Prompt Generation:**
환자의 연령(10년 단위 그룹화), 성별, 질병 상태 등의 속성을 템플릿에 넣어 문장으로 만든다 (예: "This is a brain magnetic resonance image acquired from a male with mild cognitive impairment at fifty years old"). 이 문장을 BiomedCLIP의 텍스트 인코더에 통과시켜 $(B, N, D)$ 차원의 지식 임베딩(knowledge embeddings)을 생성한다.

**Learnable Prompt Initialization:**
0으로 초기화된 학습 가능 토큰(learnable tokens)에 위에서 생성한 지식 임베딩을 더하여 초기화한다. 이를 통해 무작위 값이 아닌 생물학적 지식에서 출발하는 학습 가능 임베딩이 생성된다.

**Model Integration:**
이 프롬프트들을 인코더의 깊은 층(deep-level layers)에 입력한다. 백본 네트워크에 따라 통합 방식이 다르다.

- **U-Net / Swin UNETR:** Adaptive Average Pooling (AAP)을 통해 $(B, N, D) \rightarrow (B, N, 1)$로 변환 후, 선형 층을 거쳐 이미지 임베딩과 결합한다.
- **UNETR:** 전치(transpose) 및 선형 층을 통해 이미지 임베딩의 형상과 맞춘 뒤 결합한다.

최종적인 연산 과정은 다음과 같이 표현된다.
$$ [X_{i, \_}] = L_{i-1}([P_{i-1}, X_{i-1}]) $$
$$ [X_{i+1, \_}] = L_i([P_i, X_i]) $$
여기서 $X$는 이미지 임베딩, $P$는 지식 기반 학습 가능 임베딩, $L$은 인코더 층을 의미한다. 학습 시에는 인코더를 동결(freeze)하고, **학습 가능 프롬프트와 디코더만 업데이트**한다.

## 📊 Results

### 실험 설정

- **데이터셋:** CBMFM, ADNI, ABIDE, ABCD 등 다양한 연령대가 포함된 데이터셋 사용.
- **백본 네트워크:** U-Net (CNN), UNETR (CNN+ViT), Swin UNETR (CNN+Swin Transformer).
- **평가 지표:** Dice Similarity Coefficient (DSC), Average Surface Distance (ASD).

### 주요 결과

- **정량적 성능:** 모든 백본에서 KGPL이 가장 높은 성능을 보였으며, 특히 Swin UNETR를 사용했을 때 뇌 조직 분할 DSC 95.17%, 구조 분할 DSC 94.19%를 달성하였다.
- **파라미터 효율성:** UNETR의 경우, 전체 파인튜닝 시 92.78M 개의 파라미터를 학습시켜야 했으나, KGPL을 적용하면 단 4.46M 개의 파라미터만으로 더 높은 성능을 낼 수 있었다.
- **학습 속도:** 무작위 프롬프트를 사용한 방식보다 수렴 속도가 약 3배 빨랐다 (학습 시간 3일 $\rightarrow$ 1일).
- **정성적 분석:** 해마(hippocampus)나 대뇌 피질(cerebral cortex)과 같은 복잡한 영역에서 전체 파인튜닝이나 무작위 프롬프트 방식보다 Ground Truth에 훨씬 근접한 정밀한 분할 결과를 보여주었다.

## 🧠 Insights & Discussion

본 논문은 텍스트 기반의 생물학적 지식이 딥러닝 모델의 시각적 특징 추출을 가이드하는 강력한 도구가 될 수 있음을 입증하였다. 특히 전 생애 주기에 걸친 뇌 MRI처럼 데이터의 변동성이 크고 라벨 확보가 어려운 도메인에서, 외부의 정제된 지식(BiomedCLIP)을 프롬프트 형태로 주입하는 것이 모델의 일반화 능력을 비약적으로 상승시킨다는 점이 고무적이다.

또한, 인코더를 동결하고 소수의 프롬프트 파라미터와 디코더만 학습시킴으로써 과적합(overfitting) 위험을 줄이고 계산 효율성을 극대화하였다는 점이 강점이다.

다만, 본 연구는 현재 성인 및 청소년 데이터에 집중되어 있으며, 영유아 단계(infant phase)의 데이터셋까지 확장하여 전 생애 주기의 연속성을 완전히 확보하는 것은 향후 과제로 남아 있다.

## 📌 TL;DR

본 논문은 뇌 MRI 분할 시 연령 및 질병에 따른 해부학적 변동성 문제를 해결하기 위해, BiomedCLIP의 텍스트 임베딩을 프롬프트로 활용하는 **Knowledge-Guided Prompt Learning (KGPL)** 프레임워크를 제안한다. 이 방법은 전체 파인튜닝 대비 훨씬 적은 파라미터로도 더 높은 정확도와 빠른 수렴 속도를 보였으며, 특히 Swin UNETR 백본에서 최적의 성능을 기록하였다. 이는 의료 영상 분석에서 텍스트 기반 도메인 지식을 결합하는 것이 매우 효과적임을 시사한다.
