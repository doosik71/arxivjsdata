# EDiT: Efficient Diffusion Transformers with Linear Compressed Attention

Philipp Becker, Abhinav Mehrotra, Ruchika Chavhan, Malcolm Chadwick, Luca Morreale, Mehdi Noroozi, Alberto Gil Ramos, Sourav Bhattacharya

---

## 🧩 Problem to Solve

Diffusion Transformer (DiT)는 고품질 이미지를 생성하는 데 뛰어난 성능을 보이지만, 어텐션 메커니즘의 쿼리(Q), 키(K) 곱셈($QK^T$)에서 발생하는 이차 복잡도($O(N^2)$)는 고해상도 이미지 생성이나 자원 제약이 있는 디바이스에서의 배포를 어렵게 합니다. 특히 Multimodal DiT (MM-DiT)는 이미지와 텍스트 토큰을 함께 처리하는 Joint Attention을 사용하여 토큰 수가 증가하면서 이러한 효율성 병목 현상이 더욱 심화됩니다.

## ✨ Key Contributions

- **EDiT (Efficient Diffusion Transformer) 아키텍처 제안:** 확산 트랜스포머를 위한 새로운 선형 압축 어텐션 메커니즘을 도입했습니다.
  - **ConvFusion 함수:** 쿼리 토큰에 지역 정보를 효과적으로 통합하기 위해 다층 컨볼루션 네트워크를 사용합니다.
  - **Spatial Compressor 함수:** 키와 값 토큰의 공간 정보를 컨볼루션 기반 특징 맵을 사용하여 집계합니다.
- **MM-EDiT (Multimodal Efficient Diffusion Transformer) 아키텍처 제안:** 멀티모달 DiT를 위한 하이브리드 어텐션 방식을 개발했습니다.
  - 이미지-이미지 상호작용에는 EDiT의 선형 압축 어텐션을 적용하여 계산 효율성을 높입니다.
  - 프롬프트(텍스트) 관련 상호작용(이미지-프롬프트, 프롬프트-이미지, 프롬프트-프롬프트)에는 표준 스케일드 닷-프로덕트 어텐션을 유지합니다.
- **성능 입증:** PixArt-Σ(기존 DiT)와 Stable Diffusion 3.5-Medium(MM-DiT)에 각각 EDiT 및 MM-EDiT를 통합하고 증류(distillation)를 통해 기존 모델과 유사한 이미지 품질을 유지하면서 최대 2.2배의 속도 향상을 달성했습니다.

## 📎 Related Works

- **기존 Diffusion Transformers (DiTs):** PixArt-Σ [2], Stable Diffusion v3 [5], SANA [22] 등 최첨단 텍스트-이미지 생성 모델에 널리 사용됩니다.
- **선형 어텐션 기반 효율적인 DiTs:**
  - **Linfusion [14]:** UNet 기반 확산 모델에서 다층 비선형 변환을 사용하여 선형 어텐션의 표현력을 강화합니다.
  - **SANA [22]:** 선형 어텐션에 컨볼루션 연산을 결합하여 지역 관계를 효과적으로 포착합니다. SANA 1.5 [23]는 효율적인 스케일링을 위해 RMS 정규화를 추가합니다.
- **키-값 (KV) 토큰 압축:**
  - **PixArt-Σ [2]:** 깊이별 컨볼루션을 사용하여 키 및 값 토큰 수를 줄여 계산 효율성을 향상시켰지만, 여전히 이차 복잡성을 가집니다.
- **기타 효율적인 DiTs:**
  - **ZIGMA [9], DIMBA [6]:** Mamba [4, 7]와 같은 구조화된 상태 공간 모델(SSM)을 DiT에 통합하여 효율성을 높이려 했으나, Mamba의 인과적 특성이 복잡성을 야기하거나 이차 복잡도를 유지합니다.
- **선형 시간 MM-DiTs:**
  - **CLEAR [13]:** 희소 이웃 어텐션(sparse neighborhood attention)을 사용하여 선형 시간 복잡성을 달성하지만, 복잡한 메모리 접근 패턴과 특정 하드웨어에 대한 의존성이 있습니다.

## 🛠️ Methodology

### EDiT - 선형 압축 어텐션

EDiT는 기존 DiT의 셀프-어텐션 레이어를 대체하는 새로운 선형 압축 어텐션 메커니즘을 사용합니다.

1. **쿼리(Q) 처리 (ConvFusion):**
   - 입력 토큰 $X$로부터 쿼리 $Q_{\text{EDiT}}$를 생성하는 `ConvFusion` $\phi_{\text{CF}}(X)$ 함수를 제안합니다.
   - $\phi_{\text{CF}}(X) = \text{ReLU}(X + \text{Conv}(\text{GN}(\text{LeakyRELU}(\text{Conv}(X)))))$
   - 이 함수는 토큰 시퀀스를 원래의 잠재 이미지 형태로 재구성한 후, 2D 컨볼루션(커널 크기 $3 \times 3$의 압축 컨볼루션과 $1 \times 1$의 업샘플링 컨볼루션)을 적용하여 토큰의 지역 정보를 효과적으로 통합합니다.
2. **키(K) 및 값(V) 처리 (Spatial Compressor):**
   - 키 $K_{\text{EDiT}}$와 값 $V_{\text{EDiT}}$를 공간적으로 압축하기 위해 `Spatial Compressor` $\phi_{\text{SC}}(X)$ 함수를 도입합니다.
   - $K_{\text{EDiT}} = \text{ReLU}(\phi_{\text{SC}}(X))$ 및 $V_{\text{EDiT}} = \phi_{\text{SC}}(X)$
   - $\phi_{\text{SC}}(X) = \text{Conv}(\text{Linear}(X))$
   - 실제로는 스트라이드(stride) 2를 가진 $3 \times 3$ 깊이별 컨볼루션(depthwise convolutional kernel)을 사용하여 키와 값의 수를 4배 감소시켜 $K^T V$ 계산의 복잡도를 줄입니다.
3. **선형 어텐션 공식화:** `ConvFusion`으로 얻은 $Q_{\text{EDiT}}$와 `Spatial Compressor`로 얻은 $K_{\text{EDiT}}$, $V_{\text{EDiT}}$를 선형 어텐션 공식 ($Y_i = A_{\text{Lin}}(Q_i, K_j, V_j) = \frac{Q_i \sum_{j=1}^{N} (K_j^T V_j)}{Q_i \sum_{j=1}^{N} K_j}$)에 삽입하여 효율적인 DiT 메커니즘을 구성합니다.

### MM-EDiT - 멀티모달 DiT를 위한 하이브리드 어텐션

MM-DiT의 Joint Attention은 이미지 토큰과 프롬프트 토큰을 연결하여 셀프-어텐션을 적용하는데, 이미지 토큰 수가 압도적으로 많아 $Q_I K_I^T$ 계산이 가장 큰 병목입니다. MM-EDiT는 이를 해결하기 위해 하이브리드 어텐션을 사용합니다.

1. **Joint Attention 분해:** Joint Attention을 다음 네 가지 구성 요소로 분해합니다: (i) 이미지 토큰이 이미지 토큰에 어텐션 (image-to-image), (ii) 이미지 토큰이 프롬프트 토큰에 어텐션 (image-to-prompt), (iii) 프롬프트 토큰이 이미지 토큰에 어텐션 (prompt-to-image), (iv) 프롬프트 토큰이 자신에게 어텐션 (prompt-to-prompt).
2. **하이브리드 어텐션 적용:**
   - **이미지-이미지 어텐션 ($Q_I K_I^T$):** 계산량이 가장 많은 이 부분에 EDiT의 선형 압축 어텐션 ($A_{\text{Lin}}(Q_I, K_I, V_I)$)을 적용합니다. $Q_I$는 ConvFusion으로, $K_I, V_I$는 Spatial Compressor로 얻습니다.
   - **나머지 세 가지 상호작용 ($Q_I K_P^T, Q_P K_I^T, Q_P K_P^T$):** 원래 모델과 동일하게 표준 스케일드 닷-프로덕트 어텐션을 사용합니다. $Q_P, K_P, V_P$는 단순 선형 프로젝션을 통해 얻습니다.
3. **정규화:** 각 어텐션 블록의 출력은 정규화 계수 $\eta$를 통해 결합됩니다. 효율성을 위해 토큰 수에 기반한 근사치 $\hat{\eta}_{\text{Lin}} = \frac{N_I}{N_I + N_T}$를 사용합니다.

### 증류(Distillation) 기반 학습

EDiT와 MM-EDiT는 각각 PixArt-Σ 및 Stable Diffusion 3.5-Medium과 같은 대규모 기존 모델로부터 지식 증류(knowledge distillation)와 특징 증류(feature distillation)를 통해 학습됩니다. 이는 적은 학습 비용으로 효율적인 아키텍처가 SOTA 성능을 모방하도록 돕습니다.

## 📊 Results

- **EDiT (vs. PixArt-Σ):**
  - **정량적 결과:** $512 \times 512$ 및 $1024 \times 1024$ 해상도에서 EDiT는 teacher 모델인 PixArt-Σ와 유사하거나 더 나은 FID 및 CLIP 점수를 보였습니다. Linfusion-DiT (예: $512 \times 512$에서 8.8 Inception-v3 FID 포인트 우위) 및 SANA-DiT와 같은 기존 선형 어텐션 방식보다 우수한 성능을 입증했습니다.
  - **정성적 결과:** 생성된 이미지는 teacher 모델과 시각적으로 구별하기 어려울 정도로 고품질입니다.
  - **속도 향상:** $2048 \times 2048$ 해상도에서 EDiT는 PixArt-Σ 대비 약 2.5배 빠른 속도를 달성했습니다.
- **MM-EDiT (vs. Stable Diffusion 3.5-Medium):**
  - **정량적 결과:** SD-v3.5M과 비교하여 비슷한 FID 및 CLIP 점수를 유지하면서 SANA-MM-DiT 등 다른 선형 MM-DiT 베이스라인을 능가했습니다.
  - **속도 향상:** $1024 \times 1024$ 이미지 생성 시, 소비자용 Nvidia 3090 RTX GPU에서 SD-v3.5M 대비 2.2배 빠른 속도를 기록했습니다. Qualcomm SM8750-AB Snapdragon 8 Elite 모바일 칩셋에서도 2.2배의 런타임 단축을 보여주었습니다.
- **어블레이션 연구:** ConvFusion, Spatial Compressor, 그리고 $\hat{\eta}_{\text{Lin}}$ 근사치의 중요성을 확인했습니다. 이들의 조합이 이미지 품질과 런타임 간의 최적의 균형을 제공함을 입증했습니다.

## 🧠 Insights & Discussion

- **효율성과 품질의 균형:** EDiT와 MM-EDiT는 확산 트랜스포머의 계산 병목을 성공적으로 해결하면서도 이미지 생성 품질을 크게 희생하지 않았습니다. 이는 고해상도 이미지 생성 및 자원 제약이 있는 환경에서의 DiT 배포 가능성을 크게 확장합니다.
- **하이브리드 어텐션의 효과:** MM-EDiT에서 이미지-이미지 상호작용에 선형 어텐션을 적용하고 프롬프트 관련 상호작용에 표준 어텐션을 유지하는 하이브리드 접근 방식은 멀티모달 입력에 대한 완전 선형 어텐션 방식보다 우수함을 입증했습니다.
- **컨볼루션 기반 구성 요소의 중요성:** `ConvFusion`과 `Spatial Compressor`는 각각 지역 정보 통합 및 효율적인 토큰 압축에 핵심적인 역할을 하며, 전반적인 성능 향상에 크게 기여했습니다.
- **한계점:** 본 연구의 평가는 주로 PixArt-Σ와 Stable Diffusion 3.5-Medium 모델에 초점을 맞추었으므로, Flux [11]와 같은 더 큰 모델에 대한 일반화 가능성은 향후 연구에서 탐구될 필요가 있습니다. 또한, 정량적 지표 외에 생성 이미지에 대한 사람의 주관적인 평가도 추가적인 통찰력을 제공할 수 있습니다.
- **향후 연구 방향:** 모바일 디바이스에서 MM-EDiT의 image-to-prompt 상호작용 ($Q_I K_P^T$)이 여전히 계산 비용이 높다는 점은 개선의 여지를 남깁니다. 이는 이미지 쿼리 $Q_I$에 토큰 압축이 적용되지 않기 때문이며, MM-DiT의 하이브리드 어텐션 내에서 대체 상호작용 전략을 모색하는 것이 중요합니다.

## 📌 TL;DR

- **문제:** 기존 확산 트랜스포머(DiT)는 어텐션의 이차 복잡도로 인해 고해상도 이미지 생성 및 자원 제약 환경에서 비효율적입니다.
- **제안 방법:** `EDiT`는 `ConvFusion` (쿼리용) 및 `Spatial Compressor` (키/값용)를 활용한 선형 압축 어텐션을 도입하여 효율성을 높입니다. `MM-EDiT`는 이를 확장하여 이미지-이미지 상호작용에는 선형 어텐션을, 프롬프트 관련 상호작용에는 표준 어텐션을 사용하는 하이브리드 방식을 제안합니다. 이 모델들은 기존 대규모 DiT로부터 증류를 통해 학습됩니다.
- **주요 결과:** EDiT와 MM-EDiT는 기존 모델과 유사한 이미지 품질을 유지하면서 최대 2.2배의 상당한 속도 향상을 달성하여, 텍스트-이미지 생성의 효율성을 크게 개선했습니다.
