# Recurrent Generic Contour-based Instance Segmentation with Progressive Learning

Hao Feng, Keyi Zhou, Wengang Zhou, Yufei Yin, Jiajun Deng, Qi Sun, and Houqiang Li (2024)

## 🧩 Problem to Solve

본 논문은 인스턴스 세그멘테이션(Instance Segmentation)에서 객체의 경계선(Contour)을 효과적으로 추출하는 문제를 다룬다. 기존의 픽셀 기반 마스크(Pixel-wise mask) 예측 방식(예: Mask R-CNN)은 바운딩 박스(Bounding box)의 정확도에 의존하며, 연산 비용이 매우 높아 실시간 적용이 어렵다는 한계가 있다.

이를 해결하기 위해 제안된 Contour-based 방식들은 상대적으로 가볍지만, 다음과 같은 문제점을 가지고 있다.
1. **형태의 제약**: PolarMask와 같은 폴라 좌표계 기반 방식은 오목한(Concave) 형태의 객체를 처리하는 데 한계가 있다.
2. **학습 및 안정성 문제**: DeepSnake와 같은 데카르트 좌표계 기반 방식은 모델 크기가 크고, 반복 횟수가 증가함에 따라 오히려 성능이 저하되는 불안정성을 보인다.
3. **복잡성**: 최신 방법론들은 경계선 추정 전략이 휴리스틱하고 구조가 복잡한 경향이 있다.

따라서 본 논문의 목표는 반복적(Iterative)이고 점진적인(Progressive) 경계선 정밀화 전략을 통해, 가볍고 강건하며 다양한 형태의 객체에 적용 가능한 일반적인 Contour-based 인스턴스 세그멘테이션 네트워크인 **PolySnake**를 구축하는 것이다.

## ✨ Key Contributions

PolySnake의 핵심 아이디어는 고전적인 Snake 알고리즘에서 영감을 받아, 초기 경계선을 객체의 실제 경계로 점진적으로 변형(Deform)시키는 recurrent 구조를 설계한 것이다. 주요 기여 사항은 다음과 같다.

1. **Recurrent Update Operator**: 단일 경계선 추정치를 유지하며 이를 반복적으로 업데이트하는 recurrent 구조를 도입하여 모델을 경량화하고, 반복 횟수에 관계없이 안정적인 성능을 유지하게 한다.
2. **Multi-scale Contour Refinement (MCR)**: 고수준의 시맨틱 특징과 저수준의 세부 특징을 융합하여, 최종적으로 매우 정밀한 경계선을 도출하는 다중 스케일 정밀화 모듈을 제안한다.
3. **Shape Loss**: 객체의 기하학적 형태 학습을 강제하고 정규화하는 Shape Loss를 도입하여, 예측된 경계선이 객체 외곽에 더욱 밀착되도록 유도한다.

## 📎 Related Works

### Mask-based Instance Segmentation
Mask R-CNN과 같이 "검출 후 세그멘테이션(Detect then Segment)"하는 2단계 파이프라인이 주류를 이룬다. PANet, HTC 등이 이를 개선했으나, 여전히 바운딩 박스 의존성과 높은 연산 비용이라는 문제가 존재한다. YOLACT와 같이 박스 없이 마스크를 생성하는 방식도 있으나, 정밀한 경계선 추출에는 한계가 있다.

### Contour-based Instance Segmentation
경계선을 정점(Vertex)의 시퀀스로 예측하는 방식이다.
- **Polar-representation**: PolarMask 등은 연산이 빠르지만 복잡한 오목 형태 처리가 어렵다.
- **Cartesian-representation**: DeepSnake는 초기 경계선을 반복적으로 변형시키지만, 모델이 무겁고 반복 횟수가 늘어날수록 성능이 떨어지는 경향이 있다. E2EC는 학습 가능한 초기화 구조를 통해 성능을 높였으나, 본 논문은 여기서 더 나아가 recurrent 구조를 통한 점진적 학습과 다중 스케일 정밀화를 통해 차별점을 둔다.

## 🛠️ Methodology

PolySnake는 크게 세 가지 모듈로 구성된다: **Initial Contour Generation (ICG)**, **Iterative Contour Deformation (ICD)**, 그리고 **Multi-scale Contour Refinement (MCR)**.

### 1. Initial Contour Generation (ICG)
객체의 초기 거친 경계선 $C^0$를 생성하는 단계이다.
- **백본 네트워크**: 입력 이미지 $I$로부터 특징 맵 $F \in \mathbb{R}^{\frac{H}{R} \times \frac{W}{R} \times D}$를 추출한다.
- **세 가지 병렬 브랜치**:
    1. **Center Heatmap ($Y$)**: 객체의 중심점을 예측한다.
    2. **Offset Map ($S$)**: 중심점에서 각 정점까지의 오프셋을 예측하여 초기 경계선 정점 세트를 구성한다.
    3. **Boundary Map ($B$)**: 카테고리에 구애받지 않는 경계선 확률 지도를 예측하여, 네트워크가 경계선 특징을 더 잘 추출하도록 보조한다(학습 시에만 사용).

### 2. Iterative Contour Deformation (ICD)
초기 경계선 $C^0$를 $K$번 반복하여 정밀하게 변형시킨다.
- **Vertex-wise Feature Sampling**: 현재 정점 좌표 $C^{k-1}$에서 bilinear interpolation을 통해 특징 맵 $F$로부터 정점별 특징 $f^{k-1}$를 샘플링한다.
- **Vertex Feature Aggregation**: 정점들의 특징을 융합하여 경계선 수준의 표현 $g^{k-1}$를 생성한다. 이때 닫힌 루프 형태의 위상 구조를 반영하기 위해 **Circle-convolution**을 사용한다.
- **Vertex Coordinate Update**: GRU(Gated Recurrent Unit) 기반의 업데이트 연산자를 사용하여 정점의 변위 $\Delta C^{k-1}$를 예측한다. GRU 내부의 FC 레이어는 1-D Convolution으로 대체되었다.
    - 업데이트 식: $C^k = C^{k-1} + \Delta C^{k-1}$
    - GRU 게이트 방정식:
    $$z_k = \sigma(\text{Conv}([h_{k-1}, g_{k-1}], W_z))$$
    $$r_k = \sigma(\text{Conv}([h_{k-1}, g_{k-1}], W_r))$$
    $$\tilde{h}_k = \tanh(\text{Conv}([r_k \odot h_{k-1}, g_{k-1}], W_h))$$
    $$h_k = (1 - z_k) \odot h_{k-1} + z_k \odot \tilde{h}_k$$

### 3. Multi-scale Contour Refinement (MCR)
ICD의 결과물인 $C^K$는 다운샘플링된 특징 맵 기반이므로 고주파 세부 정보가 부족하다. 이를 보완하기 위해 FPN 구조를 사용하여 저수준 특징 맵($F^0, F^1$)을 융합한 $F'_0$를 생성하고, 이를 통해 최종 경계선 $C^M$을 도출한다.

### 4. Training Objectives
학습은 두 단계로 진행되며, 손실 함수는 다음과 같다.
- **ICG Loss**: $L_{ICG} = L_Y + L_S + L_B$ (중심점, 오프셋, 경계선 지도 손실).
- **ICD Loss**: $L_{ICD} = \sum_{k=1}^{K} \lambda^{K-k} (L_R^{(k)} + \alpha L_P^{(k)})$
    - $L_R^{(k)}$: 예측 정점과 GT 정점 간의 $\text{smooth}_{L1}$ 거리.
    - $L_P^{(k)}$ (**Shape Loss**): 인접한 정점 간의 오프셋(벡터) $\nabla C$를 계산하여, 예측된 형태와 GT 형태 간의 $\text{smooth}_{L1}$ 거리를 측정한다.
    - $\lambda < 1$인 시간적 가중치 계수를 통해 반복 횟수가 뒤로 갈수록 손실의 비중을 높인다.
- **MCR Loss**: 최종 정밀화된 경계선 $C^M$과 GT 간의 $\text{smooth}_{L1}$ 거리.

## 📊 Results

### 실험 설정
- **데이터셋**: SBD, Cityscapes, COCO, KINS(Amodal), CTW1500(Text), CULane(Lane).
- **지표**: $AP_{vol}, AP_{50}, AP_{70}$ (SBD), $AP$ (COCO, Cityscapes, KINS), $F1\text{-score}$ (Text, Lane).
- **구현**: PyTorch, Adam 옵티마이저 사용.

### 주요 결과
1. **SBD 데이터셋**: $AP_{vol}$ 기준 60.0을 달성하여, DeepSnake(54.4)와 E2EC(59.2)를 모두 앞섰다.
2. **COCO 및 Cityscapes**: DeepSnake 대비 매우 큰 성능 향상을 보였으며, 최신 SOTA인 E2EC보다도 높은 AP를 기록했다.
3. **일반화 성능 (Generalization)**: 
    - **Scene Text Detection (CTW1500)**: 곡선 텍스트를 정확하게 세그멘테이션하며 높은 정밀도(Precision)를 보였다.
    - **Lane Detection (CULane)**: 닫힌 루프가 아닌 열린 곡선 형태임에도 불구하고, Circle-convolution을 1-D Convolution으로 변경하여 적용함으로써 SOTA급 성능을 달성했다. 특히 Dazzle(눈부심) 상황에서 강건함을 보였다.
4. **효율성**: Recurrent 구조 덕분에 모델 파라미터 수가 적으며, 반복 횟수가 늘어나도 모델 크기가 증가하지 않아 실시간성에 유리하다.

## 🧠 Insights & Discussion

### 강점 및 통찰
- **Recurrent 구조의 효율성**: 기존 DeepSnake는 정밀화 모듈을 쌓을수록 파라미터가 선형적으로 증가하고 학습이 어려웠으나, PolySnake는 가중치를 공유하는 recurrent 구조를 통해 경량성을 유지하면서도 더 많은 반복 수렴이 가능하도록 설계되었다.
- **Shape Loss의 효과**: 단순히 정점의 좌표값만 맞추는 것이 아니라, 정점 간의 상대적 거리와 방향(Shape)을 학습하게 함으로써 경계선이 객체에 더 타이트하게 밀착되는 효과를 얻었다.
- **범용성**: 일반 인스턴스뿐만 아니라 텍스트, 차선과 같은 특수 형태의 곡선 추출 작업에도 쉽게 적용될 수 있음을 증명했다.

### 한계 및 비판적 해석
- **KINS 데이터셋의 낮은 향상폭**: Amodal 세그멘테이션(가려진 부분 예측) 작업인 KINS에서는 MCR 모듈의 성능 향상이 미미했다. 이는 가려진 영역의 경우 샘플링할 특징 자체가 존재하지 않기 때문에, 고해상도 특징 맵을 사용하더라도 정밀화에 한계가 있다는 점을 시사한다.
- **반복 횟수의 Trade-off**: 실험 결과 반복 횟수가 늘어날수록 성능은 향상되지만 FPS는 감소한다. 실제 적용 시에는 작업의 요구 정밀도에 따라 $K$값을 최적화하는 과정이 필수적이다.

## 📌 TL;DR

PolySnake는 recurrent 업데이트 연산자와 다중 스케일 정밀화 모듈을 결합한 경량 경계선 기반 인스턴스 세그멘테이션 모델이다. 특히 정점 간의 기하학적 구조를 학습하는 Shape Loss를 도입하여 정밀도를 높였으며, 일반 객체뿐만 아니라 곡선 텍스트, 도로 차선 검출 등 다양한 도메인에서 SOTA 성능을 입증하며 높은 범용성을 보여주었다.