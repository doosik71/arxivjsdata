# U-MixFormer: UNet-like Transformer with Mix-Attention for Efficient Semantic Segmentation

Seul-Ki Yeom, Julian von Klitzing

## 🧩 Problem to Solve

의미론적 분할(Semantic Segmentation)은 픽셀 단위의 정확한 예측을 위해 전역(global) 및 지역(local) 문맥을 균형 있게 포착하는 것이 중요합니다. 하지만 기존의 트랜스포머 기반 디코더는 종종 계산 비용이 많이 드는 특징 구성에 의존하며, 디코더 단계 간의 특징 맵 전파 효율성이 떨어지는 문제가 있습니다. 특히, 기존 방법들은 특징 맵을 개별적으로 처리하여 객체 경계 탐지를 개선할 수 있는 점진적 정제(incremental refinement) 기회를 놓치곤 합니다.

## ✨ Key Contributions

* **U-Net 기반의 새로운 디코더 아키텍처**: 효율적인 의미론적 분할을 위한 U-Net에서 영감을 받은 강력한 트랜스포머 디코더 아키텍처 U-MixFormer를 제안합니다. U-Net의 계층적 특징 캡처 및 전파 능력을 활용하여 트랜스포머 인코더의 측면 연결을 쿼리(query) 특징으로 사용하여 고수준 의미론과 저수준 구조의 조화로운 융합을 보장합니다.
* **향상된 문맥 이해를 위한 최적화된 특징 합성**: UNet-like 트랜스포머 아키텍처의 효율성을 개선하기 위해, 여러 인코더 및 디코더 출력을 키(key)와 값(value)을 위한 통합 특징으로 혼합 및 업데이트하는 `mix-attention` 메커니즘을 제안합니다. 이는 각 디코더 단계에 풍부한 특징 표현을 제공하고 문맥 이해를 향상시킵니다.
* **다양한 인코더와의 호환성**: U-MixFormer가 트랜스포머 기반(MiT, LVT) 및 CNN 기반(MSCAN) 인코더 모두와 호환됨을 입증합니다.
* **실험적 벤치마킹**: ADE20K 및 Cityscapes 데이터셋에서 계산 비용과 정확도 면에서 기존 의미론적 분할 방법들 중 새로운 SoTA(State-of-the-Art) 성능을 달성합니다. 경량, 중량, 심지어 고중량 인코더에서도 지속적으로 우수한 성능을 보입니다.

## 📎 Related Works

* **인코더 아키텍처**: SETR, PVT, Swin Transformer, SegFormer는 트랜스포머를 인코더로 활용하여 다중 스케일 특징을 생성하고 self-attention의 효율성을 높였습니다. SegNeXt와 LVT는 컨볼루션 기반 attention 메커니즘을 도입했습니다.
* **디코더 아키텍처**: DETR 및 MaskFormer 계열은 트랜스포머 디코더를 도입했지만 계산 비용이 높은 object-learnable 쿼리에 의존했습니다. FeedFormer는 인코더 특징을 직접 쿼리로 활용하여 효율성을 높였으나, 디코더 단계 간의 점진적인 특징 전파가 부족하여 객체 경계 개선 기회를 놓쳤습니다.
* **UNet-like Transformer**: TransUNet은 의료 영상 분할에 트랜스포머를 통합했고, Swin-UNet은 인코더와 디코더 모두에 Swin Transformer를 사용했습니다. U-MixFormer는 더 가벼운 디코더 단계를 사용하며, 측면 연결을 skip connection 대신 쿼리 특징으로 해석합니다.

## 🛠️ Methodology

U-MixFormer는 U-Net 구조를 기반으로 한 트랜스포머 디코더로, 효율적인 의미론적 분할을 위해 설계되었습니다.

1. **인코더 (Encoder)**:
    * 입력 이미지 $H \times W \times 3$를 처리하여 4단계의 계층적 멀티-해상도 특징 $E_i$ (i=1,...,4)를 생성합니다. 각 $E_i$는 $\frac{H}{2^{i+1}} \times \frac{W}{2^{i+1}} \times C_i$ 크기를 가집니다.

2. **디코더 (Decoder)**:
    * 인코더 단계 수와 동일한 수의 디코더 단계 $i \in \{1, ..., N\}$로 구성됩니다.
    * 각 디코더 단계 $D_{N-i+1}$는 `mix-attention` 메커니즘을 통해 정제된 특징을 순차적으로 생성합니다.
    * **쿼리 특징 ($X_q$)**: 인코더의 해당 측면 연결(lateral encoder feature map)을 쿼리 특징 $X_q^i$로 사용합니다.
    * **키/값 특징 ($X_{kv}$)**: `mix-attention` 모듈에서 다양한 인코더 및 디코더 단계의 계층적 특징 맵을 혼합하여 키와 값을 위한 통일된 표현 $X_{kv}^i$를 형성합니다.
        * 첫 번째 디코더 단계($i=1$)의 경우, 모든 인코더 특징 $\{E_j\}_{j=1}^N$이 선택됩니다.
        * 이후 단계의 경우, 이전에 계산된 디코더 단계 출력은 측면 인코더 특징을 대체하여 전파됩니다: $\{E_j\}_{j=1}^{N-i+1} \cup \{D_j\}_{j=N-i+2}^N$.
        * 선택된 특징들은 공간 차원 정렬을 위해 공간 감소 절차(spatial reduction)를 거칩니다. 이는 AvgPool과 Linear 연산을 통해 수행됩니다.
        * 공간적으로 정렬된 특징들은 채널 차원을 따라 연결되어 혼합 특징 $X_{kv}^i$를 형성합니다.
    * **디코더 블록**: Layer Normalization (LN)과 FeedForward Network (FFN)을 사용하여 `mix-attention` 모듈을 통합합니다. 기존 트랜스포머 디코더 블록에서 self-attention 모듈은 제거됩니다.
        * $$A_i = \text{LN}(\text{MixAtt.}(\text{LN}(X_{kv}^i), X_q^i)) + \text{LN}(X_q^i)$$
        * $$\text{DecoderStage}_i = D_{N-i+1} = \text{FFN}(A_i) + A_i$$

3. **최종 출력**:
    * 모든 디코더 단계의 특징 맵은 $D_1$의 높이와 너비에 맞게 Bilinear Interpolation을 사용하여 업샘플링됩니다.
    * 업샘플링된 특징들은 연결(concatenate)된 후 MLP를 통해 최종 분할 맵 $\frac{H}{4} \times \frac{W}{4} \times N_{cls}$를 예측합니다.

## 📊 Results

* **ADE20K 및 Cityscapes 벤치마크**: U-MixFormer는 모든 설정(경량, 중량, 고중량)에서 기존 SoTA 모델(SegFormer, FeedFormer, SegNeXt)을 뛰어넘는 성능을 보였습니다.
  * 예를 들어, U-MixFormer-B0는 MiT-B0 인코더를 사용하여 ADE20K에서 41.2% mIoU를 달성했으며, 이는 SegFormer-B0 및 FeedFormer-B0보다 각각 3.8% 및 2.0% 높은 mIoU와 27.3% 및 21.8% 낮은 계산 비용(GFLOPs)을 나타냅니다.
  * Cityscapes에서도 유사하게 우수한 성능을 보였으며, LVT 및 MSCAN-T/S 인코더와 결합 시에도 성능이 향상되었습니다.
* **강건성 (Robustness)**: Cityscapes-C 데이터셋(blur, noise, weather, digital 등 16가지 이미지 손상 포함)에서 SegFormer 및 FeedFormer 대비 모든 손상 범주에서 뛰어난 강건성을 입증했습니다. 특히 shot noise에서 최대 20.0-33.3%, snowy 조건에서 19.2-21.8%의 mIoU 향상을 보였습니다.
* **혼합-어텐션 및 U-Net 구조의 효과**: Mix-Attention 모듈은 mIoU를 0.7% 향상시키고 계산 비용을 줄였으며, U-Net 디코더 구조는 mIoU를 0.9% 향상시켰습니다. 이 둘을 결합한 제안된 U-MixFormer는 41.2% mIoU를 달성하여 시너지 효과를 입증했습니다.

## 🧠 Insights & Discussion

U-MixFormer는 U-Net의 계층적 특징 캡처 및 전파 능력과 트랜스포머의 전역 문맥 이해 능력을 성공적으로 융합했습니다. 특히, 측면 연결을 쿼리 특징으로 활용하고, 다양한 인코더 및 디코더 단계의 특징을 혼합하여 키와 값으로 사용하는 `mix-attention` 메커니즘은 효율적인 문맥 인코딩을 가능하게 하여 객체 경계 탐지 능력을 크게 향상시켰습니다. 이러한 접근 방식은 기존 트랜스포머 디코더가 가진 계산 효율성 및 특징 전파의 한계를 극복합니다. 또한, CNN 기반 및 트랜스포머 기반 인코더 모두와 뛰어난 호환성을 보여주며, 경량부터 고중량 모델까지 일관되게 SoTA 성능을 달성하여 그 범용성과 확장성을 입증했습니다. 다양한 이미지 손상에 대한 뛰어난 강건성은 자율 주행과 같은 안전이 중요한 애플리케이션에 U-MixFormer가 적합함을 시사합니다.

**한계 및 미래 연구**: U-MixFormer는 높은 정확도와 낮은 GFLOPs를 달성했지만, 현재 추론 시간(inference time)이 다른 경량 모델보다 느린 경향이 있습니다. 이는 U-Net 구조의 특성상 계층적 특징 유지를 위한 측면 연결에서 발생하는 오버헤드 때문입니다. 향후 가지치기(pruning) 및 지식 증류(knowledge distillation)와 같은 모델 압축 기술을 탐색하여 추론 속도를 개선할 계획입니다.

## 📌 TL;DR

U-MixFormer는 U-Net 구조와 트랜스포머의 장점을 결합한 의미론적 분할 디코더입니다. 기존 트랜스포머 디코더의 높은 계산량과 비효율적인 특징 전파 문제를 해결하기 위해, 인코더의 측면 연결을 쿼리로 사용하고 여러 인코더 및 디코더 단계의 특징을 혼합하여 키와 값으로 활용하는 `mix-attention` 메커니즘을 제안합니다. 이 방법은 계층적 특징과 전역 문맥 이해를 조화롭게 융합하여 ADE20K 및 Cityscapes 벤치마크에서 기존 SoTA 모델들을 뛰어넘는 정확도와 계산 효율성을 달성했으며, 다양한 이미지 손상에 대한 뛰어난 강건성도 입증했습니다. 다만, 추론 시간 개선은 향후 연구 과제로 남아있습니다.
