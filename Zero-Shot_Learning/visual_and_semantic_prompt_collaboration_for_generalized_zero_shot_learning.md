# Visual and Semantic Prompt Collaboration for Generalized Zero-Shot Learning

Huajie Jiang, Zhengxian Li, Xiaohan Yu, Yongli Hu, Baocai Yin, Jian Yang, Yuankai Qi (2025)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 문제는 Generalized Zero-Shot Learning (GZSL)에서 발생하는 **seen class에 대한 과적합(overfitting)**과 **시각-의미 정렬(visual-semantic alignment)의 효율성 저하** 문제이다.

GZSL은 학습 단계에서 보지 못한 unseen class를 인식하기 위해 클래스 간에 공유되는 의미 정보(semantic information)를 활용한다. 기존의 많은 접근 방식들은 시각적 특징을 의미 공간에 정렬하기 위해 visual backbone을 fine-tuning하는 방식을 사용한다. 그러나 학습 데이터(seen-class data)의 양이 제한적인 상황에서 backbone 전체를 fine-tuning할 경우, 모델이 seen class에 과도하게 최적화되어 unseen class에 대한 일반화 성능이 떨어지는 과적합 문제가 발생한다. 따라서 본 논문의 목표는 pre-trained 모델의 파라미터를 효율적으로 적응시키면서도, 시각적 정보와 의미적 정보를 효과적으로 결합하여 GZSL 성능을 높이는 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Visual Prompt와 Semantic Prompt의 협업(Collaboration)**을 통해 pre-trained Vision Transformer (ViT) 모델을 GZSL 작업에 효율적으로 적응시키는 것이다.

1. **Visual and Semantic Prompt Collaboration Network (VSPCN)**: 단순히 시각적 프롬프트만 학습하는 기존 방식과 달리, 시각적 정보 추출을 위한 visual prompt와 클래스 의미 정보를 통합하기 위한 semantic prompt를 동시에 학습하여 상호 보완적인 특징을 추출한다.
2. **계층별 차별화된 프롬프트 융합 메커니즘**: 네트워크의 얕은 층(shallow layers)에서는 기초적인 정보 통합을 위한 **Weak Prompt Fusion**을, 깊은 층(deep layers)에서는 보다 정교한 특징 추출을 위한 **Strong Prompt Fusion**을 적용하여 정보 융합의 효율성을 극대화한다.
3. **적응형 의미 정보 추출 (Semantic Adapter)**: 이미지 토큰과 의미 속성(semantic attributes) 간의 상호작용을 통해 인스턴스 수준에서 적응적인 의미 특징을 학습하는 adapter를 도입하였다.

## 📎 Related Works

### 1. Generalized Zero-Shot Learning (GZSL)

GZSL 연구는 크게 두 가지 방향으로 나뉜다.

- **Generative-based methods**: GAN, VAE, Diffusion model 등을 이용해 unseen class의 가상 시각 특징을 생성하고 이를 통해 분류기를 학습시킨다. 하지만 생성 모델의 학습이 어렵고, 생성 과정이 인식 과정과 독립적이라 효율성이 떨어진다는 한계가 있다.
- **Embedding-based methods**: 시각 공간과 의미 공간을 공통의 임베딩 공간으로 매핑하여 정렬한다. 최근에는 Attention 메커니즘을 통해 속성 관련 시각 특징을 강조하는 방식이 제안되었으나, backbone을 fine-tuning하는 과정에서 seen class에 과적합되는 문제가 여전히 존재한다.

### 2. Prompt Learning for Vision Transformer

Prompt learning은 pre-trained 모델의 가중치를 고정(frozen)하거나 일부만 튜닝하면서 소수의 학습 가능한 토큰(soft prompt)을 추가하여 downstream task에 적응시키는 효율적인 기법이다. 본 논문은 이러한 prompt tuning의 효율성을 GZSL에 접목하여, backbone의 과적합을 방지하면서도 시각-의미 정렬을 달성하고자 한다.

## 🛠️ Methodology

### 전체 시스템 구조

VSPCN은 pre-trained ViT를 backbone으로 사용하며, 입력으로 `CLS token`, `visual prompt`, `semantic prompt`, `image tokens`, `shared semantic attributes`를 받는다. 이들은 네트워크를 통과하며 계층별 융합 과정을 거쳐 최종적으로 클래스를 분류하는 데 사용된다.

### 1. Weak Prompts Fusion (얕은 층)

네트워크 초기 단계에서 무작위로 초기화된 프롬프트에 기초 정보를 주입하는 과정이다. Attention 메커니즘을 사용하여 수행된다.

- **Weak Visual Prompt Fusion (WVPF)**: 시각적 프롬프트 $f_{vp}^0$가 이미지 토큰 $F^0$로부터 중요한 시각 정보를 추출한다.
$$Q_v^0 = q(f_{vp}^0), \quad K_v^0 = k(F^0), \quad V_v^0 = v(F^0)$$
$$\tilde{f}_{vp}^0 = \delta \left( \frac{Q_v^0 (K_v^0)^T}{\sqrt{D}} \right) V_v^0$$
- **Weak Semantic Prompt Fusion (WSPF)**: 의미적 프롬프트 $f_{sp}^0$가 공유 의미 속성 $S$로부터 세밀한 속성 정보를 통합한다.
$$Q_s^0 = q(f_{sp}^0), \quad K_s^0 = k(S), \quad V_s^0 = v(S)$$
$$\tilde{f}_{sp}^0 = \delta \left( \frac{Q_s^0 (K_s^0)^T}{\sqrt{D}} \right) V_s^0$$
여기서 $\delta(\cdot)$는 softmax 연산을 의미한다.

### 2. Strong Prompts Fusion (깊은 층)

깊은 층으로 갈수록 의미 정보의 영향력이 약해지는 것을 방지하기 위해, Bias 추정 기반의 Attention을 통해 프롬프트를 업데이트한다.

- **Strong Visual Prompt Fusion (SVPF)**: 예측된 attention bias $B_v^l$을 더해 시각적 프롬프트를 업데이트한다.
$$\tilde{f}_{vp}^l = \left[ \alpha_v \delta \left( \frac{Q_v^l (K_v^l)^T}{\sqrt{D}} \right) + (1 - \alpha_v) \delta(B_v^l) \right] V_v^l + f_{vp}^l$$
- **Strong Semantic Prompt Fusion (SSPF)**: 유사한 방식으로 의미적 프롬프트를 업데이트하며, 이때 사용되는 속성 $S^l$은 아래의 Adapter를 통해 업데이트된 값이다.
$$\tilde{f}_{sp}^l = \left[ \alpha_s \delta \left( \frac{Q_s^l (K_s^l)^T}{\sqrt{D}} \right) + (1 - \alpha_s) \delta(B_s^l) \right] V_s^l + f_{sp}^l$$

- **Semantic Adapter**: 이미지 특징 $F^l$을 이용하여 속성 $S$를 인스턴스 적응형으로 업데이트한다.
$$S^l = \alpha_a \delta \left( \frac{Q_a^l (K_a^l)^T}{\sqrt{D}} \right) V_a^l + (1 - \alpha_a) S^{l-1}$$

### 3. 모델 최적화 및 손실 함수

전체 손실 함수는 다음과 같이 정의된다:
$$L = L_{BASE} + \lambda_{CED} L_{CED} + \lambda_{SKD} L_{SKD}$$

- **Base Loss ($L_{BASE}$)**: 분류 손실 $L_{CLS}$와 의미 회귀 손실 $L_{AR}$의 합으로, CLS 토큰을 클래스 의미 프로토타입 $\tilde{a}_y$에 정렬시킨다.
- **Cross-Entropy-Based Divergence Loss ($L_{CED}$)**: visual prompt가 CLS 토큰과 서로 보완적인(complementary) 판별 정보를 학습하도록 유도한다. KL-Divergence를 활용한 $L_{ED}$가 포함되어 두 토큰 간의 정보 차이를 강제한다.
- **Semantic Knowledge Distillation Loss ($L_{SKD}$)**: semantic prompt가 해당 클래스의 의미 프로토타입과 일치하도록 Jensen-Shannon Divergence와 유클리드 거리를 사용하여 학습시킨다.

### 4. 추론 (Inference)

학습 후, seen class에 편향된 예측을 보정하기 위해 명시적 보정(explicit calibration) 파라미터 $\tau$를 도입한다.
$$\tilde{y} = \arg \max_{\hat{y} \in Y} \left( f_{cls}^M \cdot a_{\hat{y}}^T + \tau I_{\hat{y} \in Y_u} \right)$$
여기서 $I_{\hat{y} \in Y_u}$는 예측 클래스가 unseen class일 때만 1이 되는 지시 함수이다.

## 📊 Results

### 실험 설정

- **데이터셋**: CUB, SUN, AWA2의 세 가지 벤치마크 데이터셋을 사용하였다.
- **평가 지표**: CZSL에서는 unseen class의 정확도(Acc)를, GZSL에서는 seen/unseen 정확도의 조화 평균인 Harmonic Mean ($H$)을 주요 지표로 사용하였다.
- **구현**: ViT-Base를 backbone으로 사용하였으며, 처음 6개 층에는 weak fusion, 이후 층에는 strong fusion을 적용하였다.

### 정량적 결과

VSPCN은 모든 데이터셋에서 기존 state-of-the-art 방법들을 능가하는 성능을 보였다.

- **CZSL**: CUB(80.6%), SUN(78.9%), AWA2(76.6%)의 정확도를 기록하여 최상위 성능을 달성하였다.
- **GZSL (Harmonic Mean)**: CUB(75.7%), SUN(53.8%), AWA2(77.6%)를 기록하며, 특히 ViT 기반의 최신 기법인 PSVMA보다 유의미하게 높은 성능 향상을 보였다.

### 분석 결과 (Ablation Study)

- **프롬프트의 영향**: visual prompt 단독 사용보다 semantic prompt 단독 사용 시 성능 향상이 더 컸으며, 두 프롬프트를 모두 사용할 때 최적의 성능을 보였다.
- **융합 메커니즘**: weak fusion과 strong fusion이 모두 존재할 때 feature adaptation이 가장 효과적으로 이루어짐을 확인하였다.
- **Adapter**: Semantic adapter가 없을 경우 성능이 하락하였으며, 이는 인스턴스 적응형 의미 특징 학습의 중요성을 시사한다.

## 🧠 Insights & Discussion

### 강점 및 기여

본 연구는 backbone 전체를 fine-tuning하는 대신 prompt tuning 방식을 채택함으로써 GZSL의 고질적인 문제인 seen class 과적합 문제를 효과적으로 완화하였다. 특히 시각적-의미적 프롬프트를 계층별로 다르게 융합하는 전략은 ViT의 깊은 층에서도 의미 정보가 소실되지 않고 유지되도록 하여, unseen class로의 지식 전이 능력을 크게 향상시켰다.

### 정성적 분석 및 해석

t-SNE 시각화 결과, VSPCN으로 추출된 특징들이 기존 CNN이나 단순 ViT-Base 대비 클래스 내 응집도(intra-class compactness)와 클래스 간 분리도(inter-class separability)가 훨씬 뛰어남이 확인되었다. 또한, Attention Map 분석을 통해 visual prompt와 semantic prompt가 각각 객체의 서로 다른 핵심 영역을 식별하며, CLS 토큰이 이 두 정보를 통합하여 가장 포괄적인 영역을 포착한다는 점을 밝혀내어 제안 방법론의 타당성을 입증하였다.

### 한계점 및 논의

논문에서는 $\tau, \lambda, \alpha$ 등 다수의 하이퍼파라미터를 사용하며, 이들의 최적값은 데이터셋마다 다르게 설정되었다(예: $\lambda_{SKD}$는 CUB에서 0.9, SUN에서 0.35). 이러한 파라미터 민감도는 실용적인 적용 시 최적화 비용을 증가시킬 수 있는 요인이다. 또한, 사용된 의미 속성(Glove) 외에 최신 LLM 기반의 더 풍부한 semantic embedding을 사용했을 때의 성능 변화에 대한 논의가 추가된다면 더욱 가치 있을 것으로 판단된다.

## 📌 TL;DR

본 논문은 GZSL에서 발생하는 seen class 과적합 문제를 해결하기 위해 **시각 및 의미 프롬프트 협업 네트워크(VSPCN)**를 제안한다. pre-trained ViT의 가중치를 고정하고, 계층별로 특화된 **Weak/Strong Prompt Fusion** 및 **Semantic Adapter**를 통해 시각 특징을 의미 공간에 효율적으로 정렬시킨다. 실험 결과, CUB, SUN, AWA2 데이터셋 모두에서 기존 SOTA 모델들을 뛰어넘는 성능을 달성하였으며, 이는 프롬프트 기반의 적응 방식이 GZSL의 지식 전이 성능을 높이는 데 매우 효과적임을 입증한다.
