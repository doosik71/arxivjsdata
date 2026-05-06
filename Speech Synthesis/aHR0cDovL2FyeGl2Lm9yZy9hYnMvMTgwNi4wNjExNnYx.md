# Stochastic WaveNet: A Generative Latent Variable Model for Sequential Data

Guokun Lai, Bohan Li, Guoqing Zheng, Yiming Yang (2018)

## 🧩 Problem to Solve

본 논문은 음성이나 인간의 필기 동작과 같은 순차적 데이터(Sequential Data)의 복잡한 확률 분포를 효과적으로 모델링하는 문제를 해결하고자 한다.

기존의 RNN, PixelCNN, WaveNet과 같은 자기회귀(Autoregressive) 모델들은 입력에서 출력으로의 결정론적인 매핑(Deterministic Mapping)을 학습한다. 이러한 모델들은 최종 출력 층에서만 확률 분포를 정의하기 때문에, 출력 분포가 단일 모드(Unimodal)이거나 단순한 혼합 모델로 제한되는 경향이 있다. 이는 실제 데이터가 가진 복잡한 분포와 차원 간의 상관관계를 충분히 포착하지 못하는 한계로 이어진다.

또한, RNN 기반의 확률적 모델(Stochastic models)들은 모델 용량을 높일 수 있지만, 순차적인 학습 구조로 인해 학습 속도가 느리다는 치명적인 단점이 있다. 따라서 본 연구의 목표는 WaveNet의 병렬 학습 가능성과 확률적 잠재 변수(Stochastic Latent Variables)의 강력한 분포 모델링 능력을 결합한 새로운 아키텍처를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 WaveNet의 계층적 구조 내의 모든 은닉 상태(Hidden states)에 확률적 잠재 변수를 주입하는 것이다.

1. **Stochastic WaveNet 구조 제안**: Dilated Convolution의 병렬 처리 이점과 잠재 변수의 유연한 분포 표현 능력을 결합하여, 순차적 데이터의 복잡한 분포를 효율적으로 학습할 수 있는 모델을 설계하였다.
2. **전용 추론 네트워크(Inference Network) 설계**: WaveNet의 특성을 반영하여, 잠재 변수의 사후 분포(Posterior distribution)를 효율적으로 추론할 수 있는 역방향 WaveNet(Reversed WaveNet) 구조를 제안하였다.
3. **Cosine KL Annealing 전략**: 다층 잠재 변수 모델의 학습 난이도를 낮추기 위해, KL 발산(KL Divergence) 항의 가중치를 코사인 함수 형태로 조절하는 스케줄링 기법을 도입하였다.

## 📎 Related Works

논문에서는 다음과 같은 관련 연구들을 언급한다.

- **결정론적 자기회귀 모델**: RNN, PixelCNN, WaveNet 등이 있으며, 높은 성능을 보이지만 출력 분포의 표현력이 제한적이다(Softmax bottleneck 현상 등).
- **RNN 기반 확률적 모델**: STORN, VRNN, SRNN, Z-forcing 등이 있다. 이들은 은닉 상태에 잠재 변수를 도입하여 표현력을 높였으나, RNN의 특성상 순차적으로 학습해야 하므로 계산 비용이 높다.
- **WaveNet 기반 변분 오토인코더(VAE)**: WaveNet을 VAE의 인코더나 디코더로 사용한 사례가 있으나, 본 논문처럼 모든 은닉 층에 잠재 변수를 직접 주입하여 시퀀스 전체의 확률 분포를 모델링하는 방식과는 차이가 있다.

## 🛠️ Methodology

### 1. 생성 모델 (Generative Model)

Stochastic WaveNet은 각 시점 $t$와 각 층 $l$에 대해 확률적 잠재 변수 $z_{t,l}$을 도입한다. 전체 결합 확률 분포는 다음과 같이 정의된다.

$$p(x,z) = \prod_{t=1}^{T} \left[ p_\theta(x_t | z_{\le t, 1:L}, x_{<t}) \prod_{l=1}^{L} p_\theta(z_{t,l} | z_{t, <l}, x_{<t}, z_{<t, 1:L}) \right]$$

여기서 $z_{t,l}$의 사전 분포(Prior)는 대각 공분산 행렬을 가진 가우시안 분포 $\mathcal{N}(z_{t,l}; \mu_{t,l}, v_{t,l})$로 정의된다. 구체적인 계산 과정은 다음과 같다.

- **은닉 상태 업데이트**: Dilated Convolution과 완전 연결 층(FC layer)을 사용하여 은닉 상태 $h_{t,l}$과 $d_{t,l}$을 계산한다.
    $$h_{t,l} = f_{\theta_1}(d_{t-2^l, l-1}, d_{t, l-1})$$
    $$d_{t,l} = f_{\theta_2}(h_{t,l}, z_{t,l})$$
- **파라미터 생성**: $h_{t,l}$을 기반으로 사전 분포의 평균 $\mu_{t,l}$과 분산 $\log v_{t,l}$을 결정하며, 최종 출력 $x_t$ 역시 은닉 표현들의 함수로 생성된다.

### 2. 변분 추론 (Variational Inference)

로그 가능도(Log-likelihood)를 직접 최대화하는 것은 불가능하므로, 변분 하한(ELBO)을 최적화한다.

$$\log p(x) \ge \mathbb{E}_{q_\phi(z|x)} \left[ \sum_{t=1}^{T} \log p_\theta(x_t | z_{\le t, 1:L}, x_{<t}) \right] - D_{KL}(q_\phi(z|x) || p_\theta(z|x))$$

**추론 네트워크 설계**:
사후 분포 $q_\phi(z|x)$를 효율적으로 계산하기 위해 **Reversed WaveNet**을 도입한다. $z_{t,l}$이 영향을 미치는 출력 집합을 $s(l,t)$라고 할 때, 미래의 정보 $x_{s(l,t)}$를 요약하는 특성 $b_{t,l}$을 다음과 같이 계산한다.

$$b_{t,l} = f_{\phi_1}(b_{t, l+1}, b_{t+2^{l+1}, l+1})$$

최종적으로 $z_{t,l}$의 사후 분포 파라미터 $\mu', \log v'$는 생성 모델의 $h_{t,l}$과 추론 모델의 $b_{t,l}$을 모두 입력으로 받아 결정된다.

### 3. 학습 전략 (KL Annealing)

학습 초기 단계에서 KL 항이 너무 강하면 잠재 변수가 데이터를 압축하는 능력을 상실하는 문제가 발생한다. 이를 해결하기 위해 KL 항에 가중치 $\lambda$를 곱하며, 본 논문에서는 기존의 선형 방식 대신 **Cosine Annealing** 전략을 사용한다.

$$\lambda_\alpha = 1 - \cos(\alpha) \quad (\alpha \in [0, \pi/2])$$

## 📊 Results

### 실험 설정

- **데이터셋**: Blizzard (영문 음성), TIMIT (음성), IAM-OnDB (필기체)
- **비교 대상(Baselines)**: RNN, VRNN, SRNN, Z-forcing, WaveNet
- **지표**: 테스트 세트의 로그 가능도(Log-likelihood) 또는 그 하한(ELBO)

### 주요 결과

1. **음성 모델링**: Blizzard와 TIMIT 데이터셋 모두에서 Stochastic WaveNet(SWaveNet)이 가장 높은 로그 가능도를 기록하며 SOTA 성능을 달성하였다. 특히 vanilla WaveNet 대비 성능 향상이 뚜렷하며, RNN 기반 확률 모델보다 학습 속도가 빠르다.
2. **필기체 생성**: IAM-OnDB 데이터셋에서 VRNN과 유사한 수준의 높은 성능을 보였으며, 생성된 샘플의 시각적 품질(글자 경계의 명확성 등)이 RNN이나 VRNN보다 우수함을 확인하였다.
3. **잠재 변수 층 수의 영향**: 실험 결과, 단일 층의 잠재 변수를 사용하는 것보다 여러 층의 계층적 구조를 가지는 것이 성능 향상에 도움이 됨을 확인하였다. 다만, 층 수가 지나치게 많아지면 각 층이 가질 수 있는 차원이 너무 작아져 오히려 성능이 하락하는 경향을 보였다.

## 🧠 Insights & Discussion

**강점**:
본 논문은 WaveNet의 연산 효율성(병렬성)을 유지하면서도, 확률적 잠재 변수를 통해 복잡한 데이터 분포를 모델링할 수 있음을 입증하였다. 특히 추론 네트워크를 역방향 WaveNet으로 설계하여 모델의 일관성을 유지하고 효율성을 높인 점이 돋보인다.

**한계 및 논의**:

- **잠재 변수 차원**: SWaveNet은 다층 구조를 가지므로 시점당 전체 잠재 변수의 차원이 RNN 기반 모델보다 큼에도 불구하고 효율적이라는 점을 주장하지만, 이에 대한 엄밀한 하이퍼파라미터 분석은 추가적으로 필요해 보인다.
- **미해결 과제**: 저자들은 향후 연구로 Z-forcing과 같은 고급 학습 전략을 SWaveNet에 적용하는 것을 제시하고 있다.

**비판적 해석**:
본 연구는 구조적인 결합(WaveNet + Latent Variables)을 통해 실질적인 성능 향상을 이끌어냈으나, 이론적인 분포의 수렴성보다는 실험적인 성능 지표에 의존하고 있다. 하지만 음성과 필기체라는 서로 다른 도메인에서 일관된 성능 향상을 보였다는 점에서 제안된 아키텍처의 범용성이 높다고 판단된다.

## 📌 TL;DR

Stochastic WaveNet은 WaveNet의 Dilated Convolution 구조에 확률적 잠재 변수를 계층적으로 주입하여, 병렬 학습의 속도와 복잡한 확률 분포 모델링 능력을 동시에 잡은 생성 모델이다. 음성 및 필기체 데이터셋에서 SOTA 성능을 기록하였으며, 이는 향후 고품질의 시퀀스 데이터 생성 및 모델링 연구에 중요한 기반이 될 수 있다.
