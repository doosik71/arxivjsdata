# Limitations of NeRF with Pre-trained Vision Features for Few-Shot 3D Reconstruction

Ankit Sanjyal (2025)

## 🧩 Problem to Solve

본 논문은 매우 제한된 수의 이미지(extreme few-shot, 5장 이하)만을 사용하여 3D 장면을 재구성하는 Few-shot 3D Reconstruction 문제에 집중한다. Neural Radiance Fields (NeRF)는 조밀한 뷰 컬렉션에서는 뛰어난 성능을 보이지만, 입력 뷰가 적은 상황에서는 기하학적 모호성으로 인해 성능이 급격히 저하되는 한계가 있다.

최근 연구들은 DINO나 CLIP과 같이 대규모 데이터셋으로 사전 학습된 Vision Feature를 NeRF에 통합하여 이러한 부족한 정보를 보완하려는 시도를 해왔다. 그러나 본 논문은 이러한 사전 학습된 특징들의 통합이 실제로 극단적인 Few-shot 시나리오에서 얼마나 효과적인지에 대해 의문을 제기하며, 이를 체계적으로 검증하는 것을 목표로 한다.

## ✨ Key Contributions

본 연구의 핵심 기여는 사전 학습된 Vision Feature(특히 DINO)를 NeRF에 결합하는 것이 Few-shot 3D 재구성 성능을 향상시킨다는 일반적인 통념과 달리, 오히려 성능을 저하시킬 수 있음을 실험적으로 입증한 것이다.

저자는 단순히 특징을 추가하는 것을 넘어, Frozen feature, LoRA(Low-Rank Adaptation) 기반의 미세 조정, 그리고 Multi-scale feature fusion이라는 단계적 접근 방식을 통해 분석을 수행하였다. 이를 통해 사전 학습된 2D 특징과 3D 재구성 작업 간의 불일치(Mismatch) 및 과적합(Overfitting) 가능성을 제시하며, 때로는 복잡한 특징 융합보다 기하학적 일관성에 집중한 단순한 구조가 더 효과적일 수 있다는 통찰을 제공한다.

## 📎 Related Works

**1. Neural Radiance Fields (NeRF):**
NeRF는 3D 좌표와 시야 방향을 색상과 밀도로 매핑하는 연속 함수를 MLP로 학습하여 고품질의 신규 뷰 합성(Novel View Synthesis)을 가능하게 하였다. Mip-NeRF나 Instant-NGP와 같은 후속 연구들이 안티앨리어싱 해결이나 학습 속도 개선을 이루었으나, 입력 뷰가 극도로 적은 상황에서의 근본적인 문제는 여전히 남아 있다.

**2. Few-Shot 3D Reconstruction:**
PixelNeRF와 같은 연구들은 이미지 특징을 NeRF의 조건(Conditioning)으로 사용하여 일반화 능력을 높이려 하였다. 또한 메타 학습(Meta-learning)이나 기하학적 사전 정보(Geometric priors)를 활용하는 방식이 제안되었으나, 본 논문이 다루는 극단적인 Few-shot 설정에서는 여전히 한계가 존재한다.

**3. Vision Features 및 Parameter-Efficient Fine-tuning:**
DINO와 CLIP 같은 모델은 강력한 시맨틱 표현력을 가지고 있어 3D 재구성 파이프라인에 도입되었다. 또한 LoRA와 같은 효율적인 미세 조정 기법을 통해 대규모 모델을 특정 작업에 맞게 최적화하려는 시도가 있었으나, 이러한 2D 특징들이 3D의 기하학적 제약 조건과 어떻게 정렬(Align)되는지에 대한 연구는 부족한 상태이다.

## 🛠️ Methodology

본 논문은 DINO 특징의 통합 수준에 따라 네 가지 변형 모델을 설계하여 성능을 비교 분석한다.

### 1. Baseline NeRF
가장 기본적인 형태로, 외부 특징 없이 좌표 기반 학습을 수행한다. 장면은 다음과 같은 함수로 모델링된다.
$$F_\theta: (\mathbf{x}, \mathbf{d}) \to (\mathbf{c}, \sigma)$$
여기서 $\mathbf{x} \in \mathbb{R}^3$는 3D 지점, $\mathbf{d} \in S^2$는 시야 방향, $\mathbf{c} \in \mathbb{R}^3$는 RGB 색상, $\sigma \in \mathbb{R}^+$는 볼륨 밀도이다. 입력값에는 고주파 성분을 캡처하기 위해 다음과 같은 Positional Encoding $\gamma(\cdot)$이 적용된다.
$$\gamma(x) = [\sin(2^0\pi x), \cos(2^0\pi x), \dots, \sin(2^{L-1}\pi x), \cos(2^{L-1}\pi x)]$$
최종 픽셀 색상 $\hat{C}(r)$은 미분 가능한 볼륨 렌더링 공식을 통해 계산된다.
$$\hat{C}(r) = \sum_{i=1}^{N} T_i \alpha_i c_i$$
단, $\alpha_i = 1 - e^{-\sigma_i \delta_i}$이며, $T_i = \prod_{j=1}^{i-1} (1 - \alpha_j)$이다.

### 2. DINO Feature Integration
사전 학습된 DINOv2-base 모델에서 추출한 특징 맵 $F_i \in \mathbb{R}^{H \times W \times D}$ ($D=768$)를 활용한다. 3D 지점 $\mathbf{x}$를 카메라 파라미터를 통해 2D 이미지 좌표 $(u_i, v_i)$로 투영한 후, Bilinear sampling을 통해 특징 $\mathbf{f}_i(\mathbf{x})$를 얻는다.

이 특징들은 학습 가능한 선형 변환을 통해 차원을 조정하며, 다음과 같이 NeRF MLP의 입력으로 결합된다.
$$f'(\mathbf{x}) = W_f \cdot \mathbf{f}(\mathbf{x}) + b_f$$
$$F_\theta([\gamma(\mathbf{x}), \gamma(\mathbf{d}), f'(\mathbf{x})]) \to (\mathbf{c}, \sigma)$$

### 3. LoRA Fine-Tuning of DINO
Frozen feature의 한계를 극복하기 위해 LoRA를 사용하여 DINO의 Attention 레이어를 효율적으로 미세 조정한다. 가중치 행렬 $W$는 다음과 같이 분해되어 업데이트된다.
$$W = W_0 + BA, \quad A \in \mathbb{R}^{r \times d}, B \in \mathbb{R}^{d \times r}$$
여기서 사전 학습된 가중치 $W_0$는 고정하고, 저차원 행렬 $A, B$ ($r=16$)만을 학습시켜 3D 재구성 작업에 최적화한다.

### 4. Multi-Scale Feature Fusion
다양한 해상도($224^2, 448^2, 896^2$)에서 특징을 추출하여 세부 디테일과 시맨틱 문맥을 동시에 캡처한다. 융합된 특징은 학습 가능한 가중치 $w_s$를 이용한 가중 합으로 계산된다.
$$\mathbf{f}_{\text{fused}}(\mathbf{x}) = \sum_{s=1}^{S} w_s \cdot \mathbf{f}^{(s)}(\mathbf{x}), \quad \sum w_s = 1$$

## 📊 Results

### 실험 설정
- **데이터셋:** NeRF Synthetic Dataset의 Lego 장면.
- **설정:** 학습 뷰 5장, 테스트 뷰 200장 (Extreme few-shot).
- **지표:** PSNR, SSIM, LPIPS.
- **학습 환경:** Adam optimizer ($\text{lr}=2 \times 10^{-4}$), 200 epoch, Apple M4 Pro (MPS backend) 사용.

### 정량적 결과
실험 결과, 모든 DINO 변형 모델이 Baseline NeRF보다 낮은 성능을 보였다.

| Method | PSNR $\uparrow$ | SSIM $\uparrow$ | LPIPS $\downarrow$ |
| :--- | :---: | :---: | :---: |
| **Baseline NeRF** | **14.71** | **0.46** | **0.53** |
| DINO-NeRF (frozen) | 12.99 | 0.46 | 0.54 |
| LoRA-NeRF (fine-tuned) | 12.97 | 0.45 | 0.54 |
| Multi-Scale LoRA-NeRF | 12.94 | 0.44 | 0.54 |

### 분석 결과
- **정성적 분석:** Baseline NeRF는 상대적으로 더 선명하고 세부적인 재구성 결과를 생성한 반면, DINO 변형 모델들은 블러링(Blurring) 현상과 아티팩트가 더 많이 관찰되었다.
- **학습 동역학:** Training PSNR 곡선 분석 결과, Baseline NeRF가 학습 전 과정에서 일관되게 더 높은 성능을 유지하였다. 이는 성능 차이가 단순히 최적화 실패로 인한 것이 아님을 시사한다.

## 🧠 Insights & Discussion

본 논문의 결과는 사전 학습된 Vision Feature가 3D 재구성, 특히 Few-shot 상황에서 항상 유익하다는 가정을 정면으로 반박한다. 성능 저하의 원인으로 다음과 같은 요인들이 분석되었다.

1. **Feature-Task Mismatch:** DINO 특징은 2D 이미지의 자기지도학습(Self-supervised learning)을 통해 학습되었다. 이는 3D 재구성에 필수적인 기하학적/광학적 제약 조건과 일치하지 않으며, 오히려 NeRF가 정확한 기하 구조를 학습하는 데 방해가 되는 노이즈나 충돌하는 그래디언트를 제공했을 가능성이 크다.
2. **Overfitting to Limited Data:** 5장의 이미지라는 극단적인 데이터 부족 상황에서, DINO 특징으로 인해 증가한 모델 복잡도가 과적합을 유발했을 수 있다. 단순한 구조의 Baseline NeRF가 오히려 새로운 시점에 대해 더 나은 일반화 성능을 보인 것으로 해석된다.
3. **Integration Challenges:** 특징 융합 과정에서 사용된 Attention 메커니즘이나 선형 투영이 2D 특징과 3D 좌표 정보를 효과적으로 결합하지 못했을 수 있다.
4. **Scale Mismatch:** Multi-scale 특징을 사용하는 것이 이론적으로는 유익하나, 실제로는 서로 다른 해상도 간의 불일치가 발생하여 오히려 재구성 품질을 해쳤을 가능성이 있다.

결론적으로, 극단적인 Few-shot 시나리오에서는 복잡한 특징 융합 방식보다 기하학적 일관성(Geometric consistency)에 집중한 단순한 아키텍처가 더 효과적일 수 있다는 점을 시사한다.

## 📌 TL;DR

본 논문은 사전 학습된 DINO 특징을 NeRF에 통합하여 Few-shot 3D 재구성 성능을 높이려는 시도가 오히려 성능을 떨어뜨린다는 사실을 체계적으로 분석하였다. 실험 결과, Baseline NeRF(PSNR 14.71)가 DINO 기반 모델들(~13.0)보다 우수함을 확인하였으며, 이는 2D 시맨틱 특징과 3D 기하학적 작업 간의 불일치 및 과적합 때문으로 분석된다. 이 연구는 향후 연구들이 단순히 강력한 2D 모델을 결합하는 것보다, 3D 특화 사전 학습이나 더 정교한 융합 메커니즘을 고민해야 함을 시사한다.