# Generalized Continual Zero-Shot Learning

Chandan Gautam, Sethupathy Parameswaran, Ashish Mishra, Suresh Sundaram (2021)

## 🧩 Problem to Solve

본 논문은 Zero-Shot Learning (ZSL)과 Continual Learning (CL)의 결합인 Continual Zero-Shot Learning (CZSL) 문제를 해결하고자 한다. 

기존의 ZSL 방식은 학습 단계에서 모든 seen class(학습 데이터가 존재하는 클래스)의 샘플을 한 번에 사용할 수 있다고 가정한다. 그러나 실제 환경에서는 데이터가 스트림 형태로 순차적으로 도착하며, 새로운 클래스가 지속적으로 추가되는 상황이 일반적이다. 이러한 환경에서 모델이 새로운 지식을 학습하면서 동시에 과거에 학습한 지식을 잊어버리는 Catastrophic Forgetting(치명적 망각) 문제와 새로운 지식을 습득하지 못하는 Intransigence(경직성) 문제가 발생한다.

따라서 본 연구의 목표는 순차적으로 도착하는 태스크(Task)로부터 학습하면서, 과거의 경험을 유지하고(CL의 목표), 학습 데이터가 없는 unseen class(미학습 클래스)까지 분류할 수 있는(ZSL의 목표) 일반화된 CZSL 프레임워크를 구축하는 것이다. 특히, 테스트 시에 어떤 태스크에 속하는지 알려주지 않는 Single-head setting(태스크 불가지론적 예측)을 구현하여 실제 적용 가능성을 높이고자 한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 생성 모델(Generative Model) 기반의 ZSL과 경험 재생(Experience Replay, ER) 및 지식 증류(Knowledge Distillation, KD)를 결합하는 것이다.

1. **경험 재생 기반의 CZSL 개발**: 소규모의 에피소드 메모리(Episodic Memory)를 사용하여 과거 태스크의 샘플을 저장하고 이를 현재 태스크와 함께 학습함으로써 망각 문제를 방지한다.
2. **Single-head 설정의 구현**: 테스트 단계에서 Task ID가 제공되지 않는 환경에서도 작동하도록 설계하여, 실전 배치에 더 적합한 일반화된 모델을 제안한다.
3. **두 가지 CZSL 실험 설정 제안**: 기존의 ZSL 방식에 가까운 설정(Setting-1)과 클래스 증분 학습(Class-incremental learning) 능력을 평가할 수 있는 설정(Setting-2)을 통해 모델의 강건성을 검증한다.
4. **다양한 생성 모델 적용**: Conditional VAE (CVAE)와 Cross and Distribution Aligned VAE (CADA)라는 두 가지 VAE 변형 모델을 CZSL 프레임워크에 적용하여 성능을 비교 분석한다.

## 📎 Related Works

### Zero-Shot Learning (ZSL)
기존 ZSL은 시각적 특징 공간과 시맨틱(속성) 공간 사이의 임베딩을 학습하는 방식(Embedding-based)이 주를 이루었으나, 이는 클래스 내부의 변동성을 포착하지 못하고 GZSL(Generalized ZSL) 환경에서 seen class로 편향되는 경향이 있다. 이를 해결하기 위해 VAE나 GAN과 같은 생성 모델을 사용하여 unseen class의 가상 특징을 생성하고, 이를 통해 ZSL 문제를 일반적인 지도 학습 문제로 변환하는 방식이 제안되었다.

### Continual Learning (CL)
CL은 새로운 데이터를 학습하면서 기존 지식을 보존하는 것을 목표로 하며, 주로 정규화 기반(Regularization-based), 메모리 기반(Memory-based), 경험 재생 기반(Experience Replay-based) 접근 방식으로 나뉜다. 본 논문은 이 중 소량의 데이터를 저장하고 재학습하는 경험 재생 방식에 집중한다.

### Continual Zero-Shot Learning (CZSL)
최근 등장한 분야로, 기존 연구들은 주로 Multi-head setting(테스트 시 태스크 ID를 제공)에 치중되어 있었다. 본 논문은 이에 대비하여 더 어려운 과제인 Single-head setting에서의 CZSL을 제안하며, 기존의 A-GEM 기반 방식과 차별화되는 생성 모델 및 경험 재생 전략을 사용한다.

## 🛠️ Methodology

### 전체 시스템 구조
본 논문은 CADA(Cross and Distribution Aligned VAE)를 기반으로 한 **CZSL-CA** 방법을 중심으로 설명한다. 전체 파이프라인은 소규모 메모리에 저장된 과거 샘플과 현재 태스크의 샘플을 함께 사용하여 인코더와 디코더를 공동 학습시키는 구조이다.

### 주요 구성 요소 및 학습 절차
1. **인코더 및 디코더**: 각 태스크 $t$에 대해 특징 인코더($E_{F t}$), 속성 인코더($E_{At}$), 특징 디코더($D_{F t}$), 속성 디코더($D_{At}$)를 학습한다.
2. **경험 재생 (Experience Replay)**: 메모리에서 샘플링된 과거 데이터와 현재 데이터를 함께 학습하여 망각을 방지한다.
3. **손실 함수 (Loss Functions)**:
    - **VAE Loss ($\mathcal{L}_{VAE}$)**: 표준 VAE의 재구성 손실과 KL 발산 손실의 합이다.
    - **Distribution-alignment Loss ($\mathcal{L}_{DA}$)**: 특징 인코더와 속성 인코더가 생성하는 잠재 공간(Latent space)의 분포를 일치시킨다.
      $$\mathcal{L}_{DA} = (\|\mu_{At}^t - \mu_{Ft}^t\|_2^2 + \|(\Sigma_{At}^t)^{1/2} - (\Sigma_{Ft}^t)^{1/2}\|_F^2)^{1/2}$$
    - **Cross-alignment Loss ($\mathcal{L}_{CA}$)**: 서로 다른 모달리티 간의 교차 재구성 손실을 최소화한다.
      $$\mathcal{L}_{CA} = |a - D_{At}^t(E_{Ft}^t(x))| + |x - D_{Ft}^t(E_{At}^t(a))|$$
    - **Knowledge Distillation Loss ($\mathcal{L}_{KD}$)**: 현재 인코더와 이전 태스크 인코더 간의 출력 차이를 줄여 지식을 전이한다.
      $$\mathcal{L}_{KD} = |E_{Ft}^t(x) - E_{Ft}^{t-1}(x)|_1 + |E_{At}^t(a) - E_{At}^{t-1}(a)|_1$$

전체 손실 함수는 다음과 같이 정의된다:
$$\mathcal{L}_{CZSL-CA} = \mathcal{L}_{VAE} + \gamma \mathcal{L}_{CA} + \delta \mathcal{L}_{DA} + \mathcal{L}_{KD}$$

### 추론 및 분류 과정
학습이 완료되면, 학습된 특징 인코더를 통해 seen class의 잠재 특징을 추출하고, 속성 인코더를 통해 unseen class의 잠재 특징을 생성한다. 이들을 결합하여 선형 분류기(Linear Classifier)를 학습시킨다. 테스트 시에는 입력 데이터를 특징 인코더로 통과시켜 잠재 특징을 얻은 후, 분류기를 통해 클래스를 예측한다.

### 샘플링 전략 (Sampling Techniques)
메모리에 저장할 샘플을 선택하는 세 가지 방법을 실험하였다:
- **Reservoir Sampling**: 입력 스트림에서 확률적으로 샘플을 무작위 선택한다.
- **Ring Buffer**: 클래스별로 동일한 크기의 큐를 유지하여 균등한 표현을 보장한다.
- **Mean of Features**: 각 클래스 특징의 이동 평균을 계산하고, 평균에 가까운 샘플을 저장한다.

## 📊 Results

### 실험 설정
- **데이터셋**: aPY, AWA1, AWA2, CUB, SUN의 5개 벤치마크 데이터셋을 사용하였다.
- **평가 지표**: Mean Seen Accuracy (mSA), Mean Unseen Accuracy (mUA), 그리고 이들의 조화 평균인 Mean Harmonic Accuracy (mH)를 측정하였다.
- **비교 대상 (Baselines)**: AGEM+CZSL, 순차적으로 학습시킨 Seq-CVAE, Seq-CADA.

### 주요 결과
1. **성능 향상**: 제안된 CZSL-CA+res(Reservoir sampling 적용) 방식이 모든 데이터셋에서 베이스라인보다 월등한 성능을 보였다. 특히 Setting-1에서 베이스라인 대비 mH가 크게 상승하였다.
2. **샘플링 전략 비교**: Reservoir sampling이 가장 좋은 성능을 냈으며, Mean of Features 방식이 가장 낮은 성능을 기록하였다.
3. **메모리 크기의 영향**: 클래스당 1~3개의 매우 적은 샘플만으로도 베이스라인보다 높은 성능을 보였으며, 샘플 수가 증가할수록 성능이 점진적으로 향상되었다.
4. **잠재 차원 분석**: 잠재 공간의 차원이 50~64일 때 최적의 성능을 보였으며, 그 이상의 차원에서는 오히려 성능이 저하되는 경향을 보였다. 이는 적절한 차원의 압축이 더 판별력 있는 특징을 생성하기 때문으로 해석된다.

## 🧠 Insights & Discussion

### 강점 및 해석
- **데이터 의존적 정규화 (Data-dependent Regularization)**: 메모리를 이용한 반복 학습이 과적합(Overfitting)을 일으키지 않고 오히려 성능을 높이는 현상이 발견되었다. 이는 $(t+1)$번째 태스크의 데이터가 $t$번째 태스크에 대한 강한 정규화 역할을 하기 때문으로 분석된다.
- **실용적 아키텍처**: Single-head 설정을 통해 태스크 ID 없이도 예측이 가능하게 함으로써 실제 환경에서의 범용성을 확보하였다.

### 한계 및 비판적 논의
- **상한선(Upper Bound)과의 격차**: 제안 방법이 베이스라인보다는 뛰어나지만, 모든 데이터를 한 번에 학습한 Offline 모델의 성능(Upper Bound)에는 여전히 미치지 못한다. 이는 순차 학습 과정에서 발생하는 불가피한 정보 손실이 존재함을 시사한다.
- **태스크 연관성**: 결과 분석에서 태스크 간의 연관성(Task relatedness)이 성능에 영향을 미친다고 언급하였으나, 구체적으로 어떤 기준(예: 시맨틱 유사도)으로 연관성을 정의하고 분석했는지에 대한 정량적 설명이 부족하다.

## 📌 TL;DR

본 논문은 데이터가 순차적으로 들어오는 환경에서 미학습 클래스까지 분류해야 하는 **Generalized Continual Zero-Shot Learning** 문제를 제안하고 해결하였다. 생성 모델(CADA/CVAE)에 **경험 재생(Experience Replay)**과 **지식 증류(Knowledge Distillation)**를 결합하여, 태스크 ID 없이도 예측 가능한 Single-head 모델을 구현하였다. 실험 결과, 클래스당 1~3개의 매우 적은 샘플만 저장하는 것만으로도 치명적 망각을 효과적으로 억제하고 ZSL 성능을 크게 향상시킬 수 있음을 증명하였다. 이 연구는 실시간으로 클래스가 확장되는 실제 시각 지능 시스템 구축에 중요한 기초가 될 수 있다.