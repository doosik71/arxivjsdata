# Diffusion-RWKV: Scaling RWKV-Like Architectures for Diffusion Models

Zhengcong Fei, Mingyuan Fan, Changqian Yu, Debang Li, Junshi Huang (2024)

## 🧩 Problem to Solve

본 연구는 고해상도 이미지 생성 작업에서 Transformer 기반 모델들이 겪는 막대한 계산 복잡도 문제를 해결하고자 한다. Transformer의 핵심인 Self-attention 메커니즘은 입력 시퀀스 길이에 대해 이차 복잡도($O(N^2)$)를 가지므로, 이미지의 해상도가 높아져 패치(patch)의 수가 증가할수록 연산 비용이 기하급수적으로 상승한다. 이러한 한계는 고해상도 이미지나 긴 비디오 생성 시 심각한 제약 요소가 된다. 따라서 본 논문의 목표는 Transformer의 강력한 표현력과 확장성(scalability)을 유지하면서도, 계산 복잡도를 선형 수준($O(N)$)으로 낮춘 새로운 Diffusion 모델 아키텍처를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 NLP 분야에서 효율성이 증명된 RWKV(Receptance Weight Key Value) 아키텍처를 Diffusion 모델의 백본으로 도입하는 것이다. 주요 기여 사항은 다음과 같다.

1. **Diffusion-RWKV 제안**: RWKV의 선형 복잡도 특성을 유지하면서 시각 데이터 처리에 최적화된 Bi-RWKV 구조를 도입하여, Transformer의 대안이 될 수 있는 저비용 고효율의 Diffusion 백본을 구축하였다.
2. **시각 데이터 최적화**: 2차원 이미지 데이터의 특성을 반영하기 위해 Quad-directional shift 연산을 적용하고, 양방향(bidirectional) 전역 어텐션을 통해 선형 복잡도 내에서 글로벌 문맥을 파악하도록 설계하였다.
3. **체계적인 설계 분석**: Conditioning 방식, 블록 설계, 파라미터 확장성 등에 대한 실험적 분석을 통해 최적의 구성(예: AdaLN-Zero, 작은 Patch size)을 도출하였다.
4. **성능 및 효율성 입증**: ImageNet 등 대규모 데이터셋에서 DiT(Diffusion Transformer)와 대등하거나 더 우수한 성능(FID)을 보이면서도, 특히 고해상도 설정에서 연산량(FLOPs)과 추론 속도를 획기적으로 개선하였다.

## 📎 Related Works

본 논문은 다음과 같은 기존 연구 흐름을 기반으로 한다.

- **Diffusion Models**: DDPM 및 LDM(Latent Diffusion Models)과 같은 생성 모델들이 이미지 합성 분야에서 표준이 되었으며, 최근에는 이를 가속화하는 샘플링 기법과 Classifier-free guidance가 널리 사용되고 있다.
- **Diffusion Architectures**: 초기에는 U-Net 구조가 주를 이루었으나, 확장성 문제로 인해 ViT 기반의 DiT(Diffusion Transformer)나 U-ViT 등이 등장하였다. 하지만 이들 역시 Self-attention의 이차 복잡도 문제에서 자유롭지 못하다.
- **Efficient Long Sequence Modeling**: Mamba와 같은 State Space Models(SSM)나 RWKV와 같은 RNN-Transformer 하이브리드 구조가 선형 복잡도로 긴 시퀀스를 처리할 수 있는 대안으로 제시되었다.

본 연구는 기존의 SSM 기반 Diffusion 연구(예: DiS)와 궤를 같이하지만, 구체적으로 RWKV 아키텍처를 Diffusion 프레임워크에 이식하고 그 확장성을 체계적으로 분석했다는 점에서 차별점을 갖는다.

## 🛠️ Methodology

### 전체 시스템 구조

Diffusion-RWKV는 이미지 입력을 패치 단위로 쪼개어 시퀀스로 변환한 뒤, 이를 적층된 Bi-RWKV 블록으로 처리하고 최종적으로 선형 디코더를 통해 노이즈를 예측하는 구조를 가진다.

### 주요 구성 요소 및 절차

1. **Image Tokenization**: 입력 이미지 $I \in \mathbb{R}^{H \times W \times C}$를 패치 크기 $p$를 이용해 $J = \frac{H \times W}{p^2}$개의 토큰으로 변환한다. 이후 학습 가능한 Positional Embedding을 추가한다.
2. **Bi-directional RWKV Block**: 기존 RWKV의 단방향성을 극복하기 위해 양방향 처리를 도입하였다.
    - **Spatial Mix (Time-Mix)**: 시퀀스 내의 의존성을 모델링하며, 2D 데이터 특성에 맞춘 Quad-directional shift와 양방향 RNN 셀을 사용한다.
    - **Channel Mix**: Time-mix의 출력을 증폭시키며, shift 연산과 활성화 함수를 통해 채널 간 정보를 융합한다.
3. **Skip Connection**: 얕은 층(shallow group)과 깊은 층(deep group)의 은닉 상태($h$)를 연결하기 위해 단순 덧셈이 아닌 Concatenation 후 Linear Projection을 적용하는 방식을 제안한다: $\text{Linear}(\text{Concate}(h_{\text{shallow}}, h_{\text{deep}}))$.
4. **Linear Decoder**: 최종 Bi-RWKV 블록의 출력을 다시 원래 이미지의 공간적 레이아웃으로 재배치하여 노이즈 $\epsilon_\theta$와 공분산을 예측한다.

### 핵심 방정식 및 연산

RWKV의 핵심인 Time-mix의 업데이트 과정은 다음과 같이 정의된다:
$$q_t = (\mu_q \odot x_t + (1-\mu_q) \odot x_{t-1}) \cdot W_q$$
$$k_t = (\mu_k \odot x_t + (1-\mu_k) \odot x_{t-1}) \cdot W_k$$
$$v_t = (\mu_v \odot x_t + (1-\mu_v) \odot x_{t-1}) \cdot W_v$$
$$o_t = (\sigma(q_t) \odot h(k_t, v_t)) \cdot W_o$$

여기서 은닉 상태 $h_t$는 다음과 같이 재귀적으로 계산된다:
$$p_t = \max(p_{t-1}, k_t)$$
$$h_t = \frac{\exp(p_{t-1}-p_t) \odot a_{t-1} + \exp(k_t-p_t) \odot v_t}{\exp(p_{t-1}-p_t) \odot b_{t-1} + \exp(k_t-p_t)}$$

### 조건부 입력 (Conditioning)

본 연구는 세 가지 방식을 비교하였으며, 최종적으로 **AdaLN-Zero**를 채택하였다. 이는 timestep $t$와 클래스 조건 $c$의 임베딩 합으로부터 scale($\gamma$)과 shift($\beta$) 파라미터를 유도하고, 잔차 연결 직전에 적용되는 scaling 파라미터 $\alpha$를 0으로 초기화하여 학습 안정성을 높이는 방식이다.

### 계산 복잡도 분석

Bi-WKV 연산의 복잡도는 다음과 같이 근사된다:
$$\text{FLOPs}(\text{Bi-WKV}(K,V)) = 13 \times J \times D$$
여기서 $J$는 토큰 수, $D$는 은닉 차원이다. 이는 Transformer의 $O(J^2 \cdot D)$에 비해 획기적인 $O(J \cdot D)$의 선형 복잡도를 가짐을 의미한다.

## 📊 Results

### 실험 설정

- **데이터셋**: CIFAR10, CelebA 64x64 (unconditional), ImageNet (class-conditional, 256x256 및 512x512 해상도).
- **평가 지표**: FID(Fréchet Inception Distance), sFID, IS(Inception Score), Precision/Recall.
- **비교 대상**: DiT, U-ViT, ADM, DiS 등.

### 주요 결과

1. **Unconditional Generation**: CIFAR10과 CelebA에서 DRWKV-S/2 모델은 U-ViT나 DiS와 대등하거나 더 우수한 FID를 기록하였으며, 특히 더 적은 파라미터 수로도 효율적인 성능을 보였다.
2. **Class-conditional Generation (ImageNet 256x256)**:
    - DRWKV-H/2 모델은 FID 2.16을 달성하여 최상위권 모델들과 경쟁 가능한 성능을 보였다.
    - DiT 대비 전체 GFLOPs를 약 25% 감소시켰다 ($1.60 \times 10^{11}$ vs $2.13 \times 10^{11}$).
3. **High-Resolution (ImageNet 512x512)**:
    - 고해상도 설정에서도 FID 2.95를 기록하며, DiS를 제외한 대부분의 CNN 및 Transformer 기반 모델을 능가하거나 대등한 성능을 보였다.
    - 해상도가 높아질수록 Transformer 대비 연산 이점이 더욱 극명하게 나타났다.

## 🧠 Insights & Discussion

### 분석 및 강점

- **Patch Size의 영향**: 실험 결과, 패치 크기가 작을수록($p=2$) FID 성능이 향상되었다. 이는 Diffusion 모델의 노이즈 예측 작업이 본질적으로 저수준(low-level) 특성을 가지므로, 세밀한 토큰화가 더 유리하기 때문으로 해석된다.
- **Conditioning 전략**: AdaLN-Zero가 In-context 방식보다 훨씬 낮은 FID를 기록하고 계산 효율성도 높았다. 이는 Diffusion 모델의 학습 안정성과 품질에 conditioning 메커니즘이 결정적인 역할을 함을 시사한다.
- **확장성(Scalability)**: 모델의 깊이($L$)와 너비($D$)를 증가시킴에 따라 FID가 일관되게 개선되는 scaling law를 확인하였으며, 이는 RWKV 구조가 대규모 파라미터 설정에서도 안정적으로 작동함을 보여준다.

### 한계 및 논의사항

- 본 연구는 주로 DiT와 비교하였으나, SiT와 같은 최신 최적화 전략을 적용한 모델과는 약간의 격차가 존재한다. 저자들은 이러한 고급 전략을 RWKV 백본에 통합하는 것을 향후 연구 과제로 남겨두었다.
- RWKV의 선형 복잡도가 주는 이점은 매우 크지만, 실제 구현 상의 병렬 처리 효율성과 메모리 대역폭 활용도에 대한 더 상세한 분석이 필요할 수 있다.

## 📌 TL;DR

본 논문은 Transformer의 이차 복잡도 문제를 해결하기 위해 **선형 복잡도 $O(N)$를 가진 RWKV 아키텍처를 도입한 Diffusion-RWKV**를 제안한다. Bi-RWKV 백본과 AdaLN-Zero, 작은 패치 사이즈($p=2$) 설정을 통해 **DiT 수준의 이미지 생성 품질을 유지하면서도 연산 비용(FLOPs)을 크게 낮추었으며**, 특히 고해상도 이미지 생성에서 압도적인 효율성을 입증하였다. 이는 향후 고해상도 멀티모달 생성 모델의 비용 효율적인 대체 아키텍처로 활용될 가능성이 매우 높다.
