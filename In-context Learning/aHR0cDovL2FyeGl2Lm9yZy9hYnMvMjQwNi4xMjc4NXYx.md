# In-Context Learning of Energy Functions

Rylan Schaeffer, Mikail Khona, Sanmi Koyejo (2024)

## 🧩 Problem to Solve

본 논문은 기존의 In-Context Learning(ICL)이 가진 표현력의 한계를 해결하고자 한다. 일반적인 ICL은 모델이 관심 대상인 In-context 분포 $p^{ICL}_\theta(x|D)$를 직접적으로 표현하거나 매개변수화할 수 있는 설정으로 제한된다. 예를 들어, 언어 모델의 경우 다음 토큰의 분포를 네트워크 출력인 logits를 통한 categorical distribution으로 표현하는 방식에 의존한다. 

이러한 방식은 조건부 확률 분포를 쉽게 매개변수화할 수 있는 상황에서만 유효하며, 이는 ICL의 범용적인 적용을 저해하는 요소가 된다. 따라서 본 연구의 목표는 특정 확률 분포의 형태에 구애받지 않고, 임의의 In-context 분포를 학습할 수 있는 보다 일반적인 형태의 ICL 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 확률 분포 $p(x|D)$를 직접 학습하는 대신, 그에 대응하는 제약 없는 임의의 에너지 함수(Energy Function) $E^{ICL}_\theta(x|D)$를 학습하는 것이다. 

에너지 기반 모델(Energy-Based Models, EBM)의 원리를 ICL에 접목하여, 모델이 입력 데이터셋 $D$가 주어졌을 때 그에 맞는 에너지 지형(Energy Landscape)을 동적으로 생성하도록 설계하였다. 특히, 이 접근 방식은 입력 공간과 출력 공간이 서로 다른(differ from one another) ICL의 첫 번째 사례를 제시함으로써, ICL이 기존에 알려진 것보다 훨씬 더 일반적인 능력을 갖추고 있음을 시사한다.

## 📎 Related Works

본 연구는 확률 모델링의 고전적인 접근법인 에너지 기반 모델(Energy-Based Models) 연구들(Hinton, 2002; Mordatch, 2018 등)에 기반하고 있다. 기존의 ICL 연구들은 주로 언어 모델링(Brown et al., 2020), 선형 회귀(Garg et al., 2022), 이미지 분류(Chan et al., 2022)와 같이 입력과 출력의 형태가 유사하거나 명확한 매개변수화가 가능한 작업에 집중되었다.

또한, 본 논문의 방법론은 Mordatch(2018)의 개념 학습(Concept Learning)과 유사한 점이 있으나, "마스크"나 "개념"에 조건화하는 대신 동일한 분포에서 추출된 데이터 시퀀스에 조건화하며, 관계 네트워크(Relational Network)를 인과적 트랜스포머(Causal Transformer)로 대체했다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 전체 시스템 구조 및 역할
본 연구에서는 Causal GPT-style Transformer를 사용하여 In-context 에너지 함수 $E^{ICL}_\theta(x|D)$를 학습한다. 모델의 파라미터 $\theta$는 고정된 상태에서, 입력으로 들어오는 데이터셋 $D$와 단일 데이터 $x$에 따라 출력 에너지 값을 적응적으로 변경한다.

기존의 트랜스포머가 각 인덱스에서 조건부 확률 분포 $p_\theta(x_n | x_{<n})$를 출력했다면, 본 모델은 각 인덱스에서 스칼라 값인 에너지 $E^{ICL}_\theta(x_n | x_{<n})$를 출력한다. 즉, 모델은 이전 $n-1$개의 데이터 포인트를 기반으로 구축된 에너지 함수를 통해 마지막 $n$번째 데이터의 에너지를 추정한다.

### 학습 목표 및 손실 함수
학습의 목적은 아래의 볼츠만 분포(Boltzmann distribution) 형태를 따르는 확률 분포를 학습하는 것이다.

$$p^{ICL}_\theta(x|D) = \frac{\exp(-E^{ICL}_\theta(x|D))}{Z_\theta}$$

여기서 $Z_\theta = \int_{x \in X} \exp(-E(x)) dx$는 분배 함수(partition function)이다. 하지만 $Z_\theta$를 직접 계산하는 것은 불가능(intractable)하므로, 본 논문에서는 Contrastive Divergence(CD) 기법을 사용하여 손실 함수를 최소화한다.

손실 함수의 그래디언트는 다음과 같이 표현되며, 이는 실제 데이터($x^+$)의 에너지는 낮추고, 모델이 생성한 가공의 데이터($x^-$)의 에너지는 높이는 방향으로 학습됨을 의미한다.

$$\nabla_\theta L(\theta) = \mathbb{E}_{p^{data}} \left[ \nabla_\theta E^{ICL}_\theta(x^+|D) \right] - \mathbb{E}_{p^{data}} \left[ \mathbb{E}_{x^- \sim p^{ICL}_\theta(x|D)} [ \nabla_\theta E^{ICL}_\theta(x^-|D) ] \right]$$

### 추론 및 샘플링 절차
학습된 에너지 함수로부터 샘플을 추출하기 위해 Langevin dynamics를 사용한다. 초기 샘플 $x_0^-$에서 시작하여, 에너지 함수의 그래디언트를 따라 에너지가 낮은(확률이 높은) 방향으로 반복적으로 업데이트한다.

$$x_{t+1}^- \leftarrow x_t^- - \alpha \nabla_x E^{ICL}_\theta(x_t^-|D) + \omega_t$$

여기서 $\alpha$는 스텝 크기이며, $\omega_t \sim \mathcal{N}(0, \sigma^2)$는 무작위 노이즈이다. 이를 통해 모델은 제약된 매개변수 형태 없이도 고확률 지점들을 샘플링할 수 있다.

## 📊 Results

### 실험 설정
- **데이터셋**: 합성 데이터인 2차원 Gaussian Mixture Model (3개의 가우시안 성분)을 사용하였다.
- **모델 구조**: 6개 레이어, 8개 헤드, 128차원 임베딩, GeLU 비선형 활성화 함수를 가진 Causal Transformer를 사용하였다.
- **학습 파라미터**: Langevin 노이즈 스케일 0.01, MCMC 스텝 15회, 스텝 크기 $\alpha = 3.16$으로 설정하여 사전 학습(Pretraining)을 수행하였다.

### 주요 결과
사전 학습 후 모델의 파라미터를 고정(freeze)한 상태에서 새로운 In-context 데이터셋 $D$가 주어졌을 때, 모델이 에너지 함수를 적응적으로 변경하는지 측정하였다. 

실험 결과, 고정된 파라미터 하에서도 In-context 데이터셋의 크기가 커질수록 에너지 지형(Energy Landscape)이 점점 더 날카롭게(sharpen) 형성되는 것이 확인되었다. 이는 모델이 파라미터 업데이트 없이 입력된 컨텍스트만을 통해 새로운 분포에 적응하는 ICL 능력을 성공적으로 수행했음을 보여준다.

## 🧠 Insights & Discussion

### 강점 및 의의
본 연구의 가장 큰 성과는 ICL의 범위를 확률 분포의 매개변수화 가능 여부라는 제약에서 해방시킨 점이다. 특히 입력 공간과 출력 공간이 서로 다른 사례를 제시함으로써, 트랜스포머가 단순히 패턴을 복제하는 것이 아니라 에너지 함수와 같은 복잡한 수학적 구조를 In-context로 학습할 수 있음을 증명하였다.

### 한계 및 비판적 해석
본 논문이 제시한 방법론의 가장 큰 한계는 학습 비용이다. 에너지 기반 모델의 특성상 학습 과정에서 네트워크 입력에 대한 미분(Differentiation w.r.t. inputs)이 필요하며, 배치당 수십에서 수백 번의 백워드 스텝이 요구된다. 이는 일반적인 트랜스포머 사전 학습에 비해 계산 비용이 훨씬 높음을 의미한다.

또한, 실험이 매우 단순한 합성 데이터셋(2D Gaussian Mixture)에서만 수행된 예비 결과(Preliminary results)라는 점이 한계이다. 실제 복잡한 고차원 데이터에서도 동일한 효율성과 적응 능력을 보일지는 추가적인 검증이 필요하다.

## 📌 TL;DR

본 논문은 특정 확률 분포의 형태에 제한받지 않는 **'에너지 함수 기반의 In-Context Learning'**을 제안한다. Causal Transformer가 스칼라 에너지 값을 출력하게 하고 Contrastive Divergence로 학습시킴으로써, 모델이 입력 데이터셋에 따라 동적으로 에너지 지형을 형성하게 만들었다. 이는 입력-출력 공간이 서로 다른 ICL의 첫 사례이며, 트랜스포머의 범용적 적응 능력을 확장했다는 점에서 의미가 있다. 향후 EBM의 높은 학습 비용 문제를 해결한다면 더 넓은 도메인에 적용될 가능성이 크다.