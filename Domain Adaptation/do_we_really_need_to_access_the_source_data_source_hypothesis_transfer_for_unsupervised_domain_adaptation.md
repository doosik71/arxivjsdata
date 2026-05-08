# Do We Really Need to Access the Source Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation

Jian Liang, Dapeng Hu, Jiashi Feng

## 🧩 Problem to Solve

기존 UDA(Unsupervised Domain Adaptation) 방법들은 레이블링된 소스 데이터셋에서 학습된 지식을 활용하여 레이블링되지 않은 타겟 도메인의 유사한 작업을 해결합니다. 그러나 대부분의 기존 UDA 방법들은 모델 적응 과정에서 소스 데이터에 직접 접근해야 하는데, 이는 분산되거나 개인 정보가 포함된 데이터에 대해 데이터 전송 비효율성 및 프라이버시 침해 위험을 야기합니다. 이 논문은 학습된 소스 모델만 사용 가능한 실용적인 환경에서 소스 데이터 없이 UDA 문제를 효과적으로 해결하는 방법을 연구합니다.

## ✨ Key Contributions

- **소스 데이터 접근 없는 UDA**: 소스 데이터에 직접 접근할 필요 없이, 사전 학습된 소스 모델(특히 분류기 모듈, 즉 가설)만을 활용하여 UDA를 수행하는 새로운 패러다임을 제안합니다. 이는 데이터 프라이버시 보호 및 효율적인 모델 전송에 기여합니다.
- **Source HypOthesis Transfer (SHOT) 프레임워크**: 소스 모델의 분류기 모듈(가설)을 고정하고, 정보 최대화(information maximization)와 자기 지도(self-supervised) 의사 라벨링(pseudo-labeling)을 활용하여 타겟 도메인 특정 특징 추출 모듈을 학습하는 간단하면서도 일반적인 표현 학습 프레임워크를 제안합니다.
- **정보 최대화(Information Maximization)**: 타겟 데이터의 특징 표현이 소스 가설에 잘 맞춰지도록 출력값을 개별적으로 확실하고 전역적으로 다양하게 만드는 $L_{ent}$ (엔트로피 최소화) 및 $L_{div}$ (다양성 증진) 손실 함수를 활용합니다.
- **자기 지도 의사 라벨링(Self-supervised Pseudo-labeling)**: 도메인 시프트로 인한 부정확하고 노이즈가 많은 의사 라벨 문제를 완화하기 위해, 타겟 도메인 자체의 중간 클래스별 프로토타입(centroids)을 생성하고 이를 통해 더 깨끗한 의사 라벨을 얻는 효과적인 방법을 제안합니다.
- **다양한 UDA 시나리오 적용**: 폐쇄형(closed-set), 부분 집합(partial-set), 개방형(open-set) 도메인 적응을 포함한 다양한 적응 시나리오에서 SHOT의 다목적성을 검증하고, 여러 벤치마크에서 SOTA(state-of-the-art) 성능을 달성합니다.
- **네트워크 아키텍처 최적화**: 레이블 스무딩(Label Smoothing), 가중치 정규화(Weight Normalization), 배치 정규화(Batch Normalization)와 같은 기술을 소스 모델 아키텍처 내에 통합하여 적응 성능을 향상시키는 방법을 모색합니다.

## 📎 Related Works

- **Unsupervised Domain Adaptation (UDA)**: 분포 모멘트(MMD) 매칭, 적대적 학습(GAN), 샘플 기반 방법, 데이터 재구성 등 다양한 접근 방식들이 연구되었으나, 대부분은 적응 과정에서 소스 데이터 접근을 필요로 합니다.
- **Hypothesis Transfer Learning (HTL)**: 학습자가 소스 데이터에 직접 접근하지 않고 소스 가설에만 의존하는 아이디어와 유사하지만, 기존 HTL은 타겟 도메인의 레이블된 데이터나 여러 소스 가설을 필요로 하여 비지도 DA에 직접 적용하기 어렵습니다.
- **Pseudo Labeling (PL)**: 준지도 학습(semi-supervised learning)에서 유래하여 비지도 DA에서도 활용되며, 최대 예측 확률을 가진 라벨로 비라벨 데이터를 라벨링하여 미세 조정(fine-tuning)에 사용됩니다. 본 논문은 DeepCluster에서 영감을 받아 자체적인 자기 지도 PL을 개발합니다.
- **Federated Learning (FL)**: 분산된 엣지 디바이스에서 데이터를 교환하지 않고 모델을 학습하는 접근 방식입니다. 최근 연합형 DA 설정이 도입되었으나, 단일 소스 도메인에만 적용되는 일반적인 DA 설정에는 한계가 있을 수 있습니다.

## 🛠️ Methodology

SHOT 프레임워크는 사전 학습된 소스 모델만을 사용하여 UDA를 수행하며, 다음 세 단계로 구성됩니다.

1. **소스 모델 생성 (Source Model Generation)**:

   - 소스 도메인 $D_s$에서 레이블링된 샘플 $\{x_i^s, y_i^s\}_{i=1}^{n_s}$을 사용하여 표준 교차 엔트로피 손실 $L_{src}$을 최소화하여 소스 함수 $f_s: X_s \to Y_s$를 학습합니다.
   - 모델의 식별력(discriminability)을 높이고 타겟 데이터 정렬을 용이하게 하기 위해 **레이블 스무딩(Label Smoothing, LS)** 기술을 도입하여 목적 함수를 $L_{ls}^{src}$로 변경합니다.
     $$L_{ls}^{src}(f_s;X_s,Y_s) = -\mathbb{E}_{(x^s,y^s)\in X_s\times Y_s} \sum_{k=1}^K q_k^{ls} \log\delta_k(f_s(x^s))$$
     여기서 $q_k^{ls} = (1-\alpha)q_k + \alpha/K$이며, $\alpha$는 스무딩 파라미터입니다.

2. **정보 최대화(Information Maximization)를 통한 소스 가설 전이 (SHOT-IM)**:

   - 소스 모델 $f_s$는 특징 인코딩 모듈 $g_s: X_s \to \mathbb{R}^d$와 분류기 모듈 $h_s: \mathbb{R}^d \to \mathbb{R}^K$로 구성됩니다 ($f_s(x) = h_s(g_s(x))$).
   - SHOT는 소스 분류기 모듈(가설) $h_s$를 고정하고 이를 타겟 분류기 모듈 $h_t$로 사용합니다 ($h_t = h_s$).
   - 타겟 도메인 특정 특징 인코딩 모듈 $g_t: X_t \to \mathbb{R}^d$를 학습하여 타겟 데이터 표현이 소스 특징 분포와 잘 일치하고 소스 가설에 의해 정확하게 분류될 수 있도록 합니다.
   - 정보 최대화(IM) 손실을 최소화하여 타겟 출력값이 개별적으로 확실하고(certain) 전역적으로 다양하게(diverse) 만듭니다. IM 손실은 엔트로피 최소화 손실 $L_{ent}$와 다양성 증진 손실 $L_{div}$로 구성됩니다.
     $$L_{ent}(f_t;X_t) = -\mathbb{E}_{x_t\in X_t} \sum_{k=1}^K \delta_k(f_t(x_t)) \log\delta_k(f_t(x_t))$$
     $$L_{div}(f_t;X_t) = \sum_{k=1}^K \hat{p}_k \log \hat{p}_k = D_{KL}(\hat{p},\frac{1}{K}\mathbf{1}_K)-\log K$$
     여기서 $f_t(x) = h_t(g_t(x))$이고, $\hat{p}=\mathbb{E}_{x_t\in X_t}[\delta(f_t(x_t))]$는 전체 타겟 도메인의 평균 출력 임베딩입니다.

3. **자기 지도 의사 라벨링(Self-supervised Pseudo-labeling)을 통한 소스 가설 전이 강화**:

   - 정보 최대화만으로는 타겟 데이터가 잘못된 소스 가설에 맞춰질 수 있는 문제를 해결하기 위해, 자기 지도 의사 라벨링을 추가합니다.
   - **클래스별 프로토타입 획득**: 이전 학습된 타겟 가설 $\hat{f}_t = \hat{g}_t \circ h_t$의 출력 $\delta_k(\hat{f}_t(x_t))$를 가중치로 사용하여 타겟 도메인 내 각 클래스의 중간 프로토타입 $c_k^{(0)}$를 계산합니다 (가중치 K-평균 군집화와 유사).
     $$c_k^{(0)} = \frac{\sum_{x_t\in X_t} \delta_k(\hat{f}_t(x_t)) \hat{g}_t(x_t)}{\sum_{x_t\in X_t} \delta_k(\hat{f}_t(x_t))}$$
   - **의사 라벨 생성**: $\hat{g}_t(x_t)$와 $c_k^{(0)}$ 사이의 코사인 거리 $D_f$를 사용하여 가장 가까운 프로토타입에 기반한 의사 라벨 $\hat{y}_t$를 할당합니다.
     $$\hat{y}_t = \arg \min_k D_f(\hat{g}_t(x_t),c_k^{(0)})$$
   - **프로토타입 정제 및 의사 라벨 갱신**: 생성된 의사 라벨을 바탕으로 프로토타입 $c_k^{(1)}$를 다시 계산하고, 이를 통해 다시 의사 라벨을 정제합니다. 실제로는 한 번의 갱신으로도 충분히 좋은 결과를 얻습니다.
   - **최종 목적 함수**: 소스 가설 $h_t=h_s$를 고정하고, 위에 생성된 의사 라벨을 활용하여 특징 인코더 $g_t$를 학습하는 전체 목적 함수는 다음과 같습니다.
     $$L(g_t) = L_{ent}(h_t \circ g_t;X_t) + L_{div}(h_t \circ g_t;X_t) - \beta \mathbb{E}_{(x_t,\hat{y}_t)\in X_t \times \hat{Y}_t} \sum_{k=1}^K \mathbf{1}_{[k=\hat{y}_t]} \log\delta_k(h_t(g_t(x_t)))$$
     여기서 $\beta > 0$는 균형을 맞추는 하이퍼파라미터입니다.

4. **소스 모델의 네트워크 아키텍처 최적화**:
   - 더 나은 소스 가설 학습을 위해 **가중치 정규화(Weight Normalization, WN)**를 최종 FC 레이어에 적용하여 각 가중치 벡터의 norm을 동일하게 유지합니다.
   - **배치 정규화(Batch Normalization, BN)**를 사용하여 내부 공변량 시프트(internal covariate shift)를 줄이고, 다른 도메인이 동일한 평균과 분산을 공유하도록 합니다.
   - 이러한 아키텍처 개선 사항들은 SHOT와 그 베이스라인 방법(소스 모델만) 모두의 성능 향상에 기여합니다.

## 📊 Results

SHOT는 다양한 UDA 시나리오와 벤치마크에서 SOTA 성능을 달성했습니다.

- **숫자 인식 (Digits dataset)**:

  - SVHN→MNIST, USPS→MNIST, MNIST→USPS 세 가지 폐쇄형 DA 작업에서 기존 SOTA 방법들을 능가하거나 경쟁적인 성능을 보였으며, 평균 정확도에서 **98.3%**로 SOTA를 달성했습니다 (SWD의 98.0% 대비).
  - SHOT (전체 모델)은 SHOT-IM보다, SHOT-IM은 소스 모델만 사용하는 경우보다 항상 더 나은 성능을 보여, 제안된 각 구성 요소의 유효성을 입증했습니다.

- **객체 인식 (Office, Office-Home, VisDA-C datasets - 폐쇄형 UDA)**:

  - **Office**: SHOT은 D→A 및 W→A와 같은 어려운 작업에서 최고 성능을 달성했으며, 평균 정확도 **88.6%**로 기존 SOTA (CDAN+TransNorm의 89.3%)에 근접한 경쟁력을 보여주었습니다.
  - **Office-Home**: 중간 규모 데이터셋에서 SHOT은 평균 정확도 **71.8%**를 기록하여 기존 SOTA (CDAN+TransNorm의 67.6%)를 크게 능가했습니다. 12개 개별 작업 중 10개에서 최고 성능을 달성했습니다.
  - **VisDA-C**: 대규모 합성-실제(synthesis-to-real) 데이터셋에서 SHOT은 클래스별 평균 정확도 **82.9%**로 SOTA를 달성했습니다 (SAFN의 76.1%, SWD의 76.4% 대비). 특히 'truck'과 같은 어려운 클래스에서 좋은 성능을 보였습니다.
  - 모든 객체 인식 벤치마크에서 SHOT (전체 모델)은 SHOT-IM보다 지속적으로 더 나은 성능을 보여 자기 지도 의사 라벨링의 중요성을 강조했습니다.

- **다양한 UDA 시나리오 (Office-Caltech, Office-Home - 폐쇄형 외)**:

  - **다중 소스(Multi-source) DA**: Office-Caltech에서 평균 **97.7%**의 정확도로 SOTA를 달성했습니다.
  - **다중 타겟(Multi-target) DA**: Office-Caltech에서 평균 **96.5%**의 정확도로 SOTA를 달성했습니다.
  - **부분 집합(Partial-set) DA (PDA)**: Office-Home에서 평균 **79.3%**의 정확도로 기존 SOTA (SAFN의 71.8%)를 넘어섰습니다.
  - **개방형(Open-set) DA (ODA)**: Office-Home에서 평균 **72.8%**의 정확도로 기존 SOTA (STA의 69.5%)를 넘어섰습니다.

- **Ablation Study**:
  - 자기 지도 의사 라벨링이 naive 의사 라벨링보다 훨씬 우수하며, $L_{ent}$와 $L_{div}$를 모두 사용하는 것이 자기 지도 PL만 사용하는 것보다 더 좋은 결과를 보여주어 다양성 증진 목표 $L_{div}$의 중요성을 확인했습니다.
  - 레이블 스무딩(LS), 가중치 정규화(WN), 배치 정규화(BN)의 각 구성 요소가 소스 모델과 SHOT-IM의 성능을 상호 보완적으로 향상시키며, 모든 구성 요소를 함께 사용할 때 SHOT-IM이 최적의 성능을 달성했습니다.

## 🧠 Insights & Discussion

- **실용적인 UDA 설정의 중요성**: 이 연구는 소스 데이터에 접근하지 않고 사전 학습된 소스 모델만으로 UDA를 수행하는 실용적인 설정의 중요성을 강조합니다. 이는 데이터 프라이버시 보호와 데이터 전송 효율성 측면에서 큰 이점을 제공합니다.
- **소스 가설 활용의 효율성**: 소스 모델의 분류기(가설)를 고정하고 타겟 도메인 특정 특징 인코더를 학습하는 전략은 도메인 적응 문제를 효과적으로 해결하며, 소스 데이터 없이도 소스 도메인의 분포 정보를 암묵적으로 활용할 수 있게 합니다.
- **정보 최대화와 자기 지도 의사 라벨링의 시너지**: 정보 최대화는 타겟 특징 표현을 소스 가설에 잘 맞추도록 유도하고, 자기 지도 의사 라벨링은 도메인 시프트로 인한 부정확한 의사 라벨 문제를 완화하여 타겟 특징 학습을 강력하게 지도합니다. 이 두 구성 요소는 상호 보완적으로 작용하여 성능을 크게 향상시킵니다.
- **아키텍처 디자인의 영향**: 레이블 스무딩, 가중치 정규화, 배치 정규화와 같은 네트워크 아키텍처 내의 요소들이 소스 모델의 품질과 결과적으로 SHOT의 적응 성능에 긍정적인 영향을 미친다는 것을 보여주었습니다.
- **다재다능함**: SHOT가 폐쇄형, 부분 집합, 개방형, 다중 소스/타겟 등 다양한 UDA 시나리오에 쉽게 확장 가능하며, 각 시나리오에서 SOTA 성능을 달성했다는 점은 이 프레임워크의 강력한 일반화 능력을 시사합니다.
- **한계**: Office 데이터셋과 같이 타겟 도메인의 크기가 매우 작은 경우에는 모델이 충분히 학습될 여지가 적어 기존 SOTA보다 성능이 약간 떨어지는 경우가 있었습니다. 이는 타겟 도메인의 데이터 규모가 SHOT의 성능에 영향을 미칠 수 있음을 보여줍니다.

## 📌 TL;DR

이 논문은 프라이버시 및 효율성 문제로 소스 데이터에 접근할 수 없을 때, 사전 학습된 소스 모델만을 활용하여 비지도 도메인 적응(UDA)을 수행하는 SHOT(Source HypOthesis Transfer) 프레임워크를 제안합니다. SHOT는 소스 분류기(가설)를 고정하고 정보 최대화와 자기 지도 의사 라벨링을 통해 타겟 도메인 특정 특징 인코더를 학습합니다. 광범위한 실험에서 SHOT는 다양한 UDA 벤치마크 및 시나리오(폐쇄형, 부분 집합, 개방형 등)에서 SOTA 성능을 달성하며, 소스 데이터 접근 없이도 효과적인 도메인 적응이 가능함을 입증했습니다.
