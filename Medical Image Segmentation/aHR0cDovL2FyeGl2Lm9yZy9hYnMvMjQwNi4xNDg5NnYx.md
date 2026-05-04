# SelfReg-UNet: Self-Regularized UNet for Medical Image Segmentation

Wenhui Zhu, Xiwen Chen, Peijie Qiu, Mohammad Farazi, Aristeidis Sotiras, Abolfazl Razi, and Yalin Wang (2024)

## 🧩 Problem to Solve

본 논문은 의료 영상 분할(Medical Image Segmentation) 분야에서 널리 사용되는 UNet 구조의 내부 학습 패턴을 분석하고, 성능을 저하시키는 두 가지 핵심 문제를 해결하고자 한다.

첫째는 **비대칭적 감독(Asymmetric Supervision)** 문제이다. UNet의 구조상 디코더(Decoder)는 최종 출력단에 가까워 정답(Ground Truth)으로부터 강한 감독 신호를 받는 반면, 인코더(Encoder)는 상대적으로 약한 신호를 받는다. 이로 인해 인코더의 일부 블록들이 분할 작업과 무관한 영역에 활성화되거나 세그멘테이션 마스크의 경계선 부분에만 집중하는 등 세만틱 손실(Semantic Loss)이 발생한다.

둘째는 **특징 중복성(Feature Redundancy)** 문제이다. 딥러닝 모델의 과매개변수화(Over-parameterization)로 인해 특징 맵(Feature Map) 내의 채널 간에 매우 유사한 정보가 중복되어 학습되는 경향이 있다. 특히 깊은 층(Deep layers)으로 갈수록 이러한 중복성이 심화되며, 이는 불필요한 계산 비용을 초래할 뿐만 아니라 작업과 무관한 시각적 특징을 학습하게 하여 성능 저하의 원인이 된다.

결과적으로 본 논문의 목표는 인코더와 디코더 간의 감독 균형을 맞추고 특징 중복성을 제거하여, 추가적인 파라미터 증가 없이 UNet 계열 모델의 분할 성능을 향상시키는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 UNet 내부에서 가장 풍부한 세만틱 정보를 가진 층을 활용해 다른 층들을 가이드하고, 채널 간의 정보 흐름을 최적화하는 **자기 정규화(Self-Regularization)** 메커니즘을 도입하는 것이다.

1. **Semantic Consistency Regularization (SCR)**: 디코더의 마지막 층($D_1$)이 가장 정확한 세만틱 정보를 보유하고 있다는 분석 결과에 기반하여, 이 층의 특징을 '교사(Teacher)'로 삼아 다른 모든 블록의 특징이 일관성을 갖도록 감독하는 방식을 제안한다.
2. **Internal Feature Distillation (IFD)**: 특징 맵의 채널을 상위 절반(Shallow)과 하위 절반(Deep)으로 나누어, 상위 채널의 유용한 컨텍스트 정보가 하위 채널로 전달되도록 유도함으로써 중복된 특징을 제거하고 표현력을 높인다.
3. **Plug-and-play 프레임워크**: 제안 방법은 추가적인 모듈 설치 없이 손실 함수(Loss Function)의 최적화만으로 구현되므로, CNN 기반 UNet뿐만 아니라 Transformer 기반 UNet(예: SwinUnet) 등 기존의 다양한 구조에 즉시 적용 가능하다.

## 📎 Related Works

기존의 UNet 개선 연구들은 주로 인코더에서 디코더로 흐르는 정보의 경로인 스킵 연결(Skip-connection)을 최적화하는 데 집중해 왔다. 예를 들어, Att-Unet는 어텐션 게이트를 통해 무관한 특징을 억제하고, UNet++는 중첩된 밀집 스킵 경로(Nested dense skip pathways)를 도입하였으며, UCTransNet은 채널 트랜스포머를 통해 스킵 연결을 재설계하였다.

하지만 이러한 연구들은 디코더에서 학습된 고수준의 세만틱 정보가 다시 인코더로 피드백되어 인코더의 학습을 돕는 방향에 대해서는 충분히 탐구하지 않았다. 본 논문은 단순히 정보의 흐름을 개선하는 것을 넘어, 디코더의 강한 감독 신호를 네트워크 전체로 확산시켜 인코더의 세만틱 손실을 막는다는 점에서 기존 접근 방식과 차별화된다.

## 🛠️ Methodology

### 전체 파이프라인

SelfReg-UNet은 기존 UNet 구조를 유지하면서 학습 단계에서 두 가지 정규화 손실 함수($L_{SCR}, L_{IFD}$)를 추가하여 모델을 최적화한다.

### 1. Semantic Consistency Regularization (SCR)

디코더의 마지막 블록($D_1$)의 특징 맵 $F_{final}$을 기준으로 다른 블록 $F^m_i$들의 세만틱 일관성을 강제한다.

- **절차**: $F_{final}$에는 평균 풀링(Average-pooling)을 적용하고, 대상 블록 $F^m_i$에는 랜덤 채널 선택(Random Channel Selection, RSC)을 적용하여 채널 및 공간 차원을 맞춘다.
- **손실 함수**: 두 특징 사이의 평균 제곱 오차(MSE)를 사용하여 다음과 같이 정의한다.
$$L_{SCR} = \frac{1}{|M-1||I|} \sum_{m=1}^{M-1} \sum_{I \sim I} \|RCS(F^m_i) - AvgPool(F_{final})\|^2$$
여기서 $M$은 전체 블록 수이며, $F^m_i$는 $D_1$을 제외한 모든 블록의 특징 맵이다.

### 2. Internal Feature Distillation (IFD)

특징 맵 내의 채널 중복성을 줄이기 위해, 채널을 상위 절반(shallow)과 하위 절반(deep)으로 분할하여 정보를 증류한다.

- **절차**: 특징 맵의 채널을 이분하여 상위 절반의 특징($F_{shallow}$)이 하위 절반의 특징($F_{deep}$)을 가이드하도록 설정한다.
- **손실 함수**: $L_p$ norm(본 논문에서는 $L_2$ norm 사용)을 이용하여 다음과 같이 정의한다.
$$L_{IFD} = \frac{1}{|M||I|} \sum_{m=1}^{M} \sum_{I \sim I} \|F_{deep} - F_{shallow}\|^p$$
이를 통해 깊은 층의 채널들이 무의미한 중복 정보를 학습하는 대신, 얕은 층의 유용한 컨텍스트를 학습하도록 유도한다.

### 3. 최종 목적 함수 (Objective Function)

전체 학습은 표준 분할 손실인 Cross-Entropy 및 Dice Loss($L_{cd}$)와 제안된 두 정규화 손실의 가중 합으로 수행된다.
$$L = L_{cd} + \lambda_1 L_{SCR} + \lambda_2 L_{IFD}$$
여기서 $\lambda_1, \lambda_2$는 각 손실의 비중을 조절하는 하이퍼파라미터이다.

## 📊 Results

### 실험 설정

- **데이터셋**: Synapse (복부 CT), ACDC (심장 MRI), GlaS (선 조직), MoNuSeg (핵 분할)의 4개 데이터셋을 사용하였다.
- **비교 대상**: Standard UNet, SwinUnet, TransUnet, Att-Unet, HiFormer, UCTransNet 등 최신 SOTA 모델들과 비교하였다.
- **평가 지표**: Dice Similarity Coefficient (DSC) 및 Intersection over Union (IoU)를 사용하였다.

### 주요 결과

- **정량적 성과**: 제안된 손실 함수를 적용했을 때 모든 데이터셋에서 성능 향상이 관찰되었다. 특히 Standard UNet에 적용했을 때 Synapse(3.49% $\uparrow$), ACDC(1.75% $\uparrow$), GlaS(5.48% $\uparrow$), MoNuSeg(3.73% $\uparrow$)의 평균 DSC 상승을 보였다.
- **어려운 타겟에 대한 강점**: Synapse 데이터셋에서 특히 담낭(Gallbladder), 왼쪽/오른쪽 신장(Kidney), 췌장(Pancreas)과 같이 분할이 어려운 장기들에서 뚜렷한 성능 향상이 나타났다.
- **정성적 분석**: GlaS 데이터셋 결과에서 기존 SOTA인 UCTransNet이 배경과 유사한 영역을 잘못 분할하거나 형태가 불완전한 반면, SelfReg-UNet(특히 SwinUnet 결합 시)은 정답(GT)에 매우 근접한 완전한 형태의 분할 결과를 생성하였다.

### 절제 연구 (Ablation Study)

- **하이퍼파라미터**: $\lambda_1 = 0.015, \lambda_2 = 0.015$일 때 최적의 성능을 보였으며, 특히 $\lambda_1$($L_{SCR}$)의 비중이 낮아질 때 성능이 빠르게 저하되는 것을 확인하여 세만틱 감독의 중요성을 입증하였다.
- **손실 함수의 유효성**: $L_{SCR}$과 $L_{IFD}$를 동시에 사용했을 때 단독으로 사용했을 때보다 더 높은 성능을 기록하여, 비대칭 감독과 특징 중복 문제가 서로 독립적으로 작용하며 둘 다 해결해야 함을 보여주었다.

## 🧠 Insights & Discussion

본 논문은 복잡한 아키텍처 수정 없이 **손실 함수 최적화만으로 모델의 내부 학습 패턴을 교정**할 수 있음을 보여주었다는 점에서 매우 효율적이다. 특히 인코더-디코더 구조의 고질적인 문제인 '감독의 불균형'을 데이터 기반의 분석(Grad-CAM 및 유사도 분석)을 통해 밝혀내고 이를 수식으로 해결한 점이 돋보인다.

**강점**:

- 추가적인 연산 비용이나 파라미터 증가가 거의 없는 'Plug-and-play' 방식이다.
- CNN과 Transformer 기반 모델 모두에 적용 가능하여 범용성이 높다.
- 모델의 해석 가능성(Interpretability) 분석을 통해 방법론의 근거를 명확히 제시하였다.

**한계 및 논의사항**:

- 하이퍼파라미터 $\lambda_1, \lambda_2$에 대한 민감도가 존재하며, 이를 최적으로 설정하기 위한 탐색 과정이 필요하다.
- 본 연구에서는 $L_2$ norm과 MSE를 사용하였으나, 논문에서 언급했듯이 KL-divergence 등 다른 지식 증류(Knowledge Distillation) 기법을 적용했을 때의 성능 차이에 대한 추가 분석이 있다면 더 견고한 결론을 낼 수 있었을 것이다.

## 📌 TL;DR

이 논문은 UNet 기반 의료 영상 분할 모델에서 발생하는 **비대칭적 감독으로 인한 세만틱 손실**과 **특징 맵의 채널 중복성** 문제를 해결하기 위해 **SelfReg-UNet**을 제안한다. 디코더의 마지막 층을 이용해 전체 네트워크의 세만틱 일관성을 맞추는 $L_{SCR}$과 채널 간 정보를 증류하는 $L_{IFD}$ 손실 함수를 도입하였으며, 이를 통해 추가 연산 비용 없이 다양한 UNet 변형 모델들의 분할 성능(DSC)을 일관되게 향상시켰다. 이 연구는 향후 더 효율적인 UNet 아키텍처 설계 및 학습 전략 수립에 중요한 기초 자료가 될 것으로 보인다.
