# $T$-former: An Efficient Transformer for Image Inpainting
Ye Deng, Siqi Hui, Sanping Zhou, Deyu Meng, Jinjun Wang

## 🧩 Problem to Solve
*   **CNN의 한계:** 기존 CNN(Convolutional Neural Networks) 기반 이미지 인페인팅(Image Inpainting) 방법들은 CNN의 지역적(local prior) 특성과 공간적으로 공유되는 정적 파라미터(spatially shared parameters) 때문에 다양하고 복잡한 형태의 손상된 이미지 처리 시 성능에 제약이 있었습니다. 특히 이미지의 장거리 의존성(long-range dependencies)을 효과적으로 포착하기 어렵습니다.
*   **Transformer의 복잡도 문제:** 최근 주목받는 어텐션 기반의 트랜스포머(Transformer) 모델은 장거리 모델링에 뛰어나지만, 표준 어텐션 연산자의 계산 복잡도가 공간 해상도에 대해 쿼드라틱($O(N^2C)$)하게 증가하여, 이미지 인페인팅과 같은 고해상도 이미지를 다루는 저수준(low-level) 비전 작업에 직접 적용하기에는 계산 비용이 너무 높다는 문제가 있습니다.

## ✨ Key Contributions
*   **선형 어텐션 메커니즘 제안:** 테일러 전개(Taylor expansion)를 활용하여 지수 함수를 근사함으로써 이미지 해상도에 대해 선형적인($O(NC^2)$) 계산 복잡도를 가지는 효율적인 어텐션 연산자를 개발했습니다. 이는 $C$ (채널)가 $N$ (공간 해상도)보다 훨씬 작다는 특성을 활용하여 효율성을 크게 개선합니다.
*   **게이팅 메커니즘 통합:** 제안된 선형 어텐션에 게이팅 메커니즘(Gating Mechanism)을 도입하여 테일러 근사로 인한 잠재적 성능 손실을 완화하고, 네트워크가 인페인팅에 유용한 특징에 집중하도록 만들었습니다.
*   **T-former 네트워크 아키텍처 설계:** 제안된 선형 어텐션 블록을 기반으로 U-Net [46] 스타일의 효율적인 이미지 인페인팅 네트워크인 $T$-former를 구축했습니다.
*   **최첨단(SOTA) 성능 달성:** 여러 벤치마크 데이터셋에서 $T$-former가 기존 최첨단 방법론들과 비교하여 유사하거나 더 우수한 성능을 달성하면서도, 파라미터 수와 계산 복잡도 면에서 효율성을 유지함을 입증했습니다.

## 📎 Related Works
*   **Vision Transformer:** 이미지 인식 [13], 객체 탐지 [6] 등 고수준 비전 작업에 트랜스포머가 도입된 이후, 데이터 효율성 [49], 특징 피라미드 [55], 로컬 어텐션 [50, 37] 등의 개선이 이루어졌습니다. 저수준 비전 작업의 고해상도 이미지 처리 문제를 해결하기 위해 저해상도 특징 처리 [15, 14, 7, 53, 62, 67]나 공간적 어텐션을 제한하는 방식 [12, 31, 56, 63, 64] 등이 제안되었습니다. 본 연구는 전체 공간 픽셀 간 어텐션을 유지하면서 복잡도를 줄이는 접근 방식을 취합니다.
*   **Image Inpainting:**
    *   **비학습 기반:** 초기에는 확산(diffusion-based) [1, 3, 4, 9] 및 예시(exemplar-based) [2, 5, 28, 58] 방식이 주로 사용되었으나, 작은 손상이나 단순 패턴에만 효과적이었습니다.
    *   **학습 기반 (CNNs):** GAN(Generative Adversarial Network) [17] 프레임워크를 도입하여 조건부 이미지 생성 문제로 해결하는 방식 [24, 30, 42, 66]이 주류가 되었습니다. 컨볼루션의 정적 파라미터 문제를 해결하기 위해 수동 [34] 또는 자동 [57, 61]으로 특징을 조정하는 컨볼루션 변형이나, 엣지 [40], 구조 [18, 29, 35, 45], 의미 [32, 33] 등 추가 정보를 활용한 안내 방식이 연구되었습니다.
    *   **학습 기반 (Attention):** 컨텍스트 어텐션(contextual attention) [36, 54, 59, 60, 65]이 도입되어 장거리 의존성 모델링에 기여했으나, 높은 계산 부담으로 인해 네트워크에 대규모 배포가 어려웠습니다. 본 논문의 선형 어텐션은 이 계산 부담 문제를 해결합니다.

## 🛠️ Methodology
*   **선형 어텐션 구현:**
    *   **바닐라 어텐션의 한계 극복:** 표준 어텐션은 $N$이 공간 해상도 $H \times W$일 때 $O(N^2C)$의 복잡도를 가집니다. 이를 극복하기 위해, 소프트맥스 연산의 핵심인 지수 함수 $\exp(x)$를 테일러 전개를 통해 $\exp(x) \approx 1+x$로 근사합니다.
    *   **행렬 곱셈 순서 변경:** 어텐션 계산식인 $O = \text{Softmax}(\frac{QK^{\top}}{\sqrt{C}})V$에서 $(QK^{\top})V$ 대신 $Q(K^{\top}V)$ 순서로 계산 순서를 변경합니다. 이로 인해 계산 복잡도는 $O(N C^2)$로 줄어듭니다. 이미지 인페인팅에서 채널 $C$는 공간 해상도 $N$에 비해 훨씬 작기 때문에, 이는 큰 효율성 개선으로 이어집니다.
    *   **잔차 항 추가:** 테일러 근사에서 비롯된 잔차 항($V+$)을 어텐션 결과에 추가하여 성능을 더욱 향상시킵니다. 이는 `V` (value) 행렬을 직접 더하는 형태로 구현됩니다.
*   **게이팅 메커니즘 (LAG: Linear Attention with Gating mechanism):**
    *   제안된 선형 어텐션 연산자 $\text{A}(\cdot)$의 출력 $A$에 게이팅 값 $G$를 Hadamard 곱($\odot$)하여 최종 출력 $O = A \odot G$를 계산합니다.
    *   게이팅 값 $G$는 입력 특징 $X$에 $1 \times 1$ 컨볼루션과 GELU [20] 활성화 함수를 적용하여 얻습니다. 이는 테일러 근사로 인해 발생할 수 있는 "부정확한" 어텐션 출력을 조정하여, 네트워크의 후속 레이어가 인페인팅에 기여하는 유용한 특징에 집중하도록 돕습니다.
*   **네트워크 아키텍처 ($T$-former):**
    *   **U-Net 스타일 구조:** 인코더-디코더 구조를 기반으로 하며, 인코더와 디코더 간 스킵 연결(skip connection)을 통해 다중 스케일 특징을 융합합니다.
    *   **트랜스포머 블록:** $T$-former의 핵심 빌딩 블록으로, LAG 레이어와 단순한 FFN (Feed-Forward Network)으로 구성됩니다. 각 서브 레이어에는 잔차 연결(residual connection)이 적용됩니다.
    *   **인코더:** 마스크가 적용된 이미지 $I_m \in \mathbb{R}^{3 \times H \times W}$를 입력으로 받아 $7 \times 7$ 컨볼루션을 거친 후, 4단계의 인코더 스테이지를 통과합니다. 각 스테이지는 여러 트랜스포머 블록으로 구성되며, 단계 사이에 $3 \times 3$ 스트라이드 2 컨볼루션을 사용하여 특징 맵을 다운샘플링합니다.
    *   **디코더:** 인코더의 최종 특징 맵을 입력으로 받아 3단계의 디코더 스테이지를 통해 고해상도 이미지 표현을 점진적으로 복원합니다. 각 단계에서 최근접 이웃 보간과 $3 \times 3$ 컨볼루션을 통한 업샘플링이 이루어지며, 인코더의 특징과 스킵 연결로 연결된 후 $1 \times 1$ 컨볼루션을 통해 채널이 조정됩니다.
    *   **FFN 설계:** FFN은 $1 \times 1$ 컨볼루션과 $3 \times 3$ 깊이별 컨볼루션(depth-wise convolution)의 조합을 사용하여 복잡도를 줄였습니다.
*   **손실 함수:**
    *   총 손실 $L = \lambda_r L_{re} + \lambda_p L_{perc} + \lambda_s L_{style} + \lambda_a L_{adv}$를 사용합니다. 각 계수는 $\lambda_r=1, \lambda_p=1, \lambda_s=250, \lambda_a=0.1$로 설정됩니다.
    *   **재구성 손실 ($L_{re}$):** 출력 $I_{out}$과 그라운드 트루스(ground truth) $I_g$ 간의 $L_1$ 거리입니다: $L_{re} = \|I_{out} - I_g\|_1$.
    *   **지각 손실 ($L_{perc}$):** ImageNet [11]에 사전 학습된 VGG-19 [48] 네트워크의 중간 레이어 특징 맵에서 계산됩니다.
    *   **스타일 손실 ($L_{style}$):** VGG-19 특징 맵에서 구성된 Gram 행렬 간의 $L_1$ 거리입니다.
    *   **적대적 손실 ($L_{adv}$):** Spectral Normalization [39]이 적용된 PatchGAN 판별자 [69]를 사용하여 이미지 분포의 사실성을 높입니다.

## 📊 Results
*   **데이터셋 및 설정:** Paris street view [42], CelebA-HQ [26], Places2 [68] 데이터셋에서 평가되었습니다. 모든 이미지는 $256 \times 256$으로 크기가 조정되었고, PC [34]의 마스크 데이터셋을 사용하여 다양한 마스크 비율(10-20%, 20-30%, 30-40%, 40-50%)로 손상 정도를 시뮬레이션했습니다.
*   **평가 지표:** FID (Fréchet Inception Distance, $\downarrow$, 낮을수록 좋음), PSNR (Peak Signal-to-Noise Ratio, $\uparrow$, 높을수록 좋음), SSIM (Structural Similarity Index, $\uparrow$, 높을수록 좋음)을 사용했습니다.
*   **정량적 비교:**
    *   $T$-former는 모든 데이터셋과 다양한 마스크 비율에 걸쳐 기존 SOTA 모델(GC [61], RFR [30], CTN [12], DTS [18]) 대비 FID는 가장 낮고, PSNR과 SSIM은 가장 높거나 대등한 최상위 성능을 달성했습니다.
    *   특히, $T$-former는 51.3G MACs와 14.8M 파라미터로, 비교 대상 모델들보다 훨씬 낮은 계산 복잡도와 파라미터 수를 가짐에도 불구하고 우수한 성능을 보여주었습니다 (예: RFR 206.1G/30.6M, DTS 75.9G/52.1M).
*   **정성적 비교:**
    *   기존 모델들은 종종 흐릿함(GC, CTN), 명백한 아티팩트 및 의미 불일치(RFR), 복잡한 패턴에서 구조적 비일관성(DTS)과 같은 문제를 보였습니다.
    *   이에 반해 $T$-former는 대부분의 경우 더 합리적이고 사실적인 이미지를 복원하여, 시각적으로도 우수함을 입증했습니다.
*   **Ablation Study (Paris 데이터셋):**
    *   **V (테일러 전개 잔차 항):** 이 잔차 항을 제거하면 FID, PSNR, SSIM 지표가 악화되었습니다. 특히 이미지 손상 정도가 클수록 잔차 항의 기여가 더 중요해지는 것으로 나타났습니다.
    *   **G (게이팅 메커니즘):** 게이팅 메커니즘을 제거했을 때도 성능 저하가 관찰되었습니다. 이는 게이팅 메커니즘이 선형 어텐션의 "부정확성"을 보정하고 유용한 특징에 집중하게 하여 인페인팅 품질을 높이는 데 필수적임을 시사합니다.
    *   두 요소(V 및 G)를 모두 제거했을 때 가장 큰 성능 저하가 발생했습니다.

## 🧠 Insights & Discussion
*   **효율적인 장거리 의존성 모델링:** $T$-former는 기존 CNN 기반 인페인팅 모델이 가지던 장거리 의존성 모델링의 한계를 효과적으로 극복했습니다. 동시에 표준 트랜스포머의 높은 계산 복잡도를 혁신적인 선형 어텐션으로 해결하여, 고해상도 이미지 인페인팅 작업에 트랜스포머를 실용적으로 적용할 수 있는 길을 열었습니다.
*   **테일러 근사의 실용성:** 테일러 전개를 이용한 선형 어텐션은 채널 차원이 공간 해상도에 비해 훨씬 작다는 이미지 특성을 활용하여 계산 효율성을 극대화했습니다. 이는 낮은 수준의 비전 작업에서 트랜스포머의 활용 가능성을 확장합니다.
*   **게이팅 메커니즘과 잔차 항의 시너지:** 게이팅 메커니즘은 테일러 근사로 인한 "부정확성"을 보정하여 네트워크가 유효한 특징에 집중하게 돕고, 잔차 항($V+$)은 특히 손상이 심한 경우 모델의 복원력을 강화하는 앙상블 효과와 유사한 역할을 합니다. 이 두 가지 요소가 결합하여 $T$-former의 견고하고 우수한 성능에 기여합니다.
*   **균형 잡힌 성능:** $T$-former는 최첨단 성능을 달성하면서도 상대적으로 적은 파라미터와 계산 비용을 유지하여, 실용적인 이미지 인페인팅 애플리케이션에 매우 적합함을 보여줍니다.

## 📌 TL;DR
본 논문은 CNN의 장거리 모델링 한계와 트랜스포머의 고해상도 이미지 처리 복잡도 문제를 해결하기 위해, 테일러 전개 기반의 효율적인 선형 어텐션과 게이팅 메커니즘을 제안합니다. 이를 핵심 블록으로 활용하여 U-Net 스타일의 $T$-former 네트워크를 구축했으며, $T$-former는 낮은 계산 비용으로 이미지 인페인팅 벤치마크에서 최첨단 성능을 달성하여 더 사실적이고 시각적으로 일관된 이미지 복원 결과를 제공합니다.