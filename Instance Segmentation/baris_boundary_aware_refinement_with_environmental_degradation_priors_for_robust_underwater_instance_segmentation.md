# BARIS: Boundary-Aware Refinement with Environmental Degradation Priors for Robust Underwater Instance Segmentation

Pin-Chi Pan, Soo-Chang Pei (2025)

## 🧩 Problem to Solve

본 논문은 수중 환경에서 발생하는 고유한 시각적 왜곡으로 인해 발생하는 수중 인스턴스 분할(Underwater Instance Segmentation)의 성능 저하 문제를 해결하고자 한다. 수중 환경은 빛의 감쇠(light attenuation), 산란(scattering), 그리고 파장으로 인한 색상 왜곡(color distortion)과 같은 열악한 시각적 조건이 존재하며, 이는 객체의 경계를 모호하게 만들고 이미지 품질을 저하시킨다. 또한, 해양 눈(marine snow)이라 불리는 부유 입자와 수면 반사 등이 세부 정보의 손실과 오분류를 유발한다.

이러한 이유로 지상 환경 데이터로 학습된 기존의 분할 모델을 수중 데이터셋에 적용할 경우, 텍스처, 조명, 투명도 등의 차이로 인해 성능이 크게 떨어진다. 따라서 본 연구의 목표는 수중 환경의 열악한 특성을 효과적으로 모델링하고, 모호한 객체 경계를 정밀하게 복원하여 강건한 수중 인스턴스 분할을 수행하는 프레임워크를 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 수중 이미지의 환경적 특성을 반영하는 **어댑터 기반의 튜닝**과 **다단계 특성 정밀화**, 그리고 **경계 인식 손실 함수**를 결합하여 정밀도를 높이는 것이다. 주요 기여 사항은 다음과 같다.

1. **BARIS-Decoder**: 다단계 게이트 정밀화 네트워크(MSGRN)와 깊이별 분리 업샘플링(DSU)을 통해 다중 스케일 특성을 정밀하게 융합함으로써 마스크 예측의 정밀도를 향상시킨다.
2. **Environmental Robust Adapter (ERA)**: 수중 환경의 열화 패턴을 학습하는 경량 어댑터를 도입하여, 전체 파라미터를 미세 조정(full fine-tuning)하는 것보다 90% 이상 적은 파라미터만으로도 수중 환경에 효율적으로 적응하게 한다.
3. **Boundary-Aware Cross-Entropy (BACE) Loss**: 선형 대수의 Range-Null Space 분해 개념을 적용하여, 전역적 구조는 유지하면서 모호한 경계 부분의 세부 사항을 정밀하게 복원하는 새로운 손실 함수를 제안한다.

## 📎 Related Works

기존의 수중 이미지 분할 연구는 주로 다중 스케일 특성 융합(Multi-scale feature fusion)이나 어댑터 기반 튜닝(Adapter-based tuning)에 집중해 왔다.

- **다중 스케일 융합 방식**: WaterMask 및 RefineMask와 같은 방법론은 FPN(Feature Pyramid Networks) 구조를 통해 다양한 해상도의 공간 정보를 집계하여 특성 표현을 강화했다. 그러나 이러한 방식은 문맥적 이해는 높여주지만, 객체의 경계를 명시적으로 정밀화하는 메커니즘이 부족하여 물고기 떼와 같이 객체가 밀집된 복잡한 장면에서 경계 오류가 발생하는 한계가 있다.
- **어댑터 기반 튜닝 방식**: USIS-SAM은 Vision Transformer(ViT)에 수중 사전 지식을 통합하는 어댑터를 사용하여 환경 적응력을 높였다. 하지만 기존 어댑터 기술들은 주로 특성 변조(feature modulation)에 집중할 뿐, 수중의 산란이나 색상 왜곡과 같은 물리적 열화 현상을 직접적으로 해결하거나 경계의 모호성을 처리하는 명시적 장치가 부족하다.

BARIS는 이러한 한계를 극복하기 위해 환경 열화 사전 지식(environmental degradation priors)을 직접 모델링하는 ERA와 경계 정밀도에 특화된 BACE Loss를 도입하여 차별점을 둔다.

## 🛠️ Methodology

### 1. BARIS-Decoder

BARIS-Decoder는 모든 백본 단계($F_1$부터 $F_4$)의 다중 스케일 특성을 집계하여 공간적 정밀도를 높인다. 최종 마스크 $M_{out}$은 다음과 같이 생성된다.
$$M_{out} = \text{Conv}_{1\times1}(\Phi(F_1, F_2, F_3, F_4))$$
여기서 $\Phi(\cdot)$는 BARIS-Decoder이며, 다음의 반복적인 정밀화 단계를 거친다.
$$\hat{F}_4^i = \text{M}_{\text{MSGRN}}^i(F_1, F_2, F_3, F_4^i)$$
$$F_4^{i+1} = \text{M}_{\text{DSU}}^i(\hat{F}_4^i)$$

#### 1.1 Multi-Stage Gated Refinement Network (MSGRN)

MSGRN은 **Multi-Scale Gated Attention (MSGAttention)**을 사용하여 정보가 많은 영역은 강조하고 중복되는 영역은 억제한다.

- 먼저 깊이별 분리 합성곱(DSConv)과 ROIAlign을 통해 특성을 추출하고 결합(Concat)한다.
- MSGAttention은 다음과 같은 연산을 통해 특성 가중치를 동적으로 조절한다.
$$\hat{Z} = \text{MSGAttention}(X, X'_1) = \sigma_{\text{sig}}(W) \odot (Y \odot V)$$
여기서 $\odot$은 Hadamard product(원소별 곱)이며, 이를 통해 국부적인 특성을 정밀하게 다듬어 경계 정확도를 높인다.

#### 1.2 Depthwise Separable Upsample (DSU)

DSU는 단순한 보간법 대신 다중 스케일 깊이별 합성곱과 **Pixel Shuffle**을 결합하여 고해상도 복원을 수행한다.

- $3\times3, 5\times5, 7\times7$ 크기의 서로 다른 커널을 가진 DWConv를 통해 다양한 수용 영역(receptive field)에서 고주파 세부 정보를 추출한다.
- 추출된 특성들의 평균을 구한 뒤 Pixel Shuffle을 적용하여 연산 비용을 낮추면서도 공간적 일관성을 유지하며 해상도를 높인다.

### 2. ERA-Tuning (Environmental Robust Adapter)

ERA는 수중 이미지의 열화 패턴을 캡처하기 위해 설계된 경량 튜닝 모듈이다.

#### 2.1 Multi-Scale Feature Extraction (MSFE)

다양한 커널 크기의 DWConv와 Max-pooling을 사용하여 공간 표현력을 강화하며, Channel Attention (CA) 메커니즘을 통해 중요한 스펙트럼 정보를 동적으로 재가중치화한다.

#### 2.2 Environmental Adaptation

학습 가능한 환경 임베딩 $E \in \mathbb{R}^{N \times C}$를 사용하여 $N$가지의 수중 환경 조건(예: 탁도, 빛 흡수 등)을 모델링한다.

- 각 픽셀에 대해 환경 기술자(environmental descriptor)를 계산한다.
$$E_{\text{adapted}} = \sigma_{\text{soft}}(\text{Linear}(F_c)) \otimes E$$
- 이후 가중치 게이팅 메커니즘을 통해 특성을 변조한다.
$$F_e = \phi(F_c \odot \sigma_{\text{sig}}(E_{\text{adapted}}))$$
최종적으로 zero-initialized up-projection을 통해 초기 학습을 안정화하며 백본 특성에 더해준다.

### 3. Boundary-Aware Cross-Entropy (BACE) Loss

BACE Loss는 선형 대수의 **Range-Null Space Decomposition** 개념을 활용한다.

- **Range Space**: 이미지의 지배적인 구조(저주파 성분)를 유지한다.
- **Null Space**: 누락된 고주파 세부 사항(경계선)을 캡처한다.

정밀화된 예측값 $\Gamma$는 다음과 같이 정의된다.
$$\Gamma(M_\theta, M_{gt}) = A^T A M_{gt} + (I - A^T A) M_\theta$$
여기서 $A$는 Max-pooling 연산자(구조 추출), $A^T$는 Nearest-neighbor interpolation(세부 복원)을 의미한다. 즉, 정답 마스크($M_{gt}$)에서는 전역 구조를 가져오고, 예측 마스크($M_\theta$)에서는 세부 경계 정보를 가져와 결합함으로써 더 날카롭고 정확한 경계를 생성한다.

최종 손실 함수는 표준 교차 엔트로피($L_{CE}$)와 BACE Loss의 가중 합으로 구성된다.
$$L_{\text{Total}} = L_{CE}(M_\theta, M_{gt}) + \lambda \cdot L_{\text{BACE}}(M_\theta, M_{gt})$$

## 📊 Results

### 1. 실험 설정

- **데이터셋**: UIIS (Underwater Image Instance Segmentation) 및 USIS10K.
- **백본**: Swin Transformer (Swin-B) 및 ConvNeXt V2-B.
- **지표**: mAP, $AP_{50}$, $AP_{75}$ 등.
- **비교 대상**: Mask R-CNN, Cascade Mask R-CNN, PointRend, SOLOv2, Mask2Former, WaterMask, USIS-SAM.

### 2. 주요 결과

- **정량적 성능**: UIIS 데이터셋에서 BARIS-ERA는 Swin-B 백본 기준 mAP 31.6%를 달성하여 Mask R-CNN(28.2%) 대비 3.4%p 향상되었으며, ConvNeXt V2-B에서는 mAP 32.3%를 기록하여 SOTA 성능을 달성했다.
- **효율성**: ERA-tuning은 전체 파라미터를 미세 조정하는 방식에 비해 학습 가능한 파라미터 수를 90% 이상 줄이면서도(Swin-B 기준 약 4.67%만 사용), 성능은 오히려 Full Fine-tuning(mAP 28.2%)보다 높은 mAP 29.9%를 기록했다.
- **경계 정밀도**: 정성적 분석 결과, BARIS-ERA는 탁도가 높거나 조명이 왜곡된 환경에서도 다른 모델보다 객체 경계를 훨씬 더 정밀하게 획득하였으며, 특히 밀집된 객체 영역에서 과분할(over-segmentation) 현상을 크게 줄였다.

## 🧠 Insights & Discussion

### 1. 강점 및 분석

- **상호 보완적 설계**: BARIS-Decoder는 특성 융합을 통해 정밀도를 높이고, ERA는 환경적 왜곡을 해결하며, BACE Loss는 경계를 다듬는다. 이 세 가지 구성 요소가 서로 보완적으로 작용하여 수중 분할의 난제들을 체계적으로 해결했다.
- **환경 적응력의 시각화**: t-SNE 분석 결과, ERA를 적용한 특성 분포가 사전 학습된 지상 데이터의 분포와 밀접하게 겹치는 것을 확인하였다. 이는 ERA가 수중 특성을 지상 특성 공간으로 효과적으로 정렬(align)시켜 모델의 적응력을 높였음을 의미한다.

### 2. 한계 및 향후 과제

- **극한 환경의 어려움**: 매우 심한 탁도나 극단적인 조명 변화가 있는 극한의 수중 환경에서는 여전히 경계 획득이 어렵다는 점이 언급되었다.
- **실시간성**: BARIS-ERA는 높은 정확도를 제공하지만, 다단계 정밀화 과정으로 인해 기본 Mask R-CNN보다는 추론 속도(FPS)가 다소 낮다. 향후 실시간 애플리케이션을 위한 추론 효율화 연구가 필요하다.

## 📌 TL;DR

본 논문은 수중 이미지의 열화 현상을 극복하고 정밀한 인스턴스 분할을 수행하기 위한 **BARIS-ERA** 프레임워크를 제안한다. 핵심은 **(1) 다단계 게이트 네트워크 기반의 BARIS-Decoder**, **(2) 환경 열화 사전 지식을 학습하는 경량 어댑터 ERA**, **(3) Range-Null Space 분해 기반의 BACE Loss**이다. 이를 통해 매우 적은 파라미터 튜닝만으로도 수중 환경에 강건하게 적응하며, 특히 객체의 경계 정밀도를 획기적으로 높여 수중 인스턴스 분할 분야에서 SOTA 성능을 달성하였다. 이 연구는 수중 로봇이나 환경 모니터링 시스템의 시각 인지 능력을 향상시키는 데 크게 기여할 것으로 기대된다.
