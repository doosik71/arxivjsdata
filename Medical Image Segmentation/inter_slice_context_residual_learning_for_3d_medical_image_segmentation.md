# Inter-slice Context Residual Learning for 3D Medical Image Segmentation

Jianpeng Zhang, Yutong Xie, Yan Wang, and Yong Xia (2020)

## 🧩 Problem to Solve

3D 의료 영상 분할(Medical Image Segmentation)은 질병의 진행 상태를 평가하고 치료 계획을 수립하는 데 필수적이다. 하지만 의료 영상의 특성상 연조직의 대비(contrast)가 낮고, 장기나 종양의 모양, 크기, 위치가 매우 다양하여 정확한 분할이 매우 어렵다. 또한, 전문적인 주석(annotation) 데이터의 부족으로 인해 모델 학습에 제약이 따른다.

최근 3D Deep Convolutional Neural Networks(DCNNs)가 등장하며 슬라이스 간의 문맥 정보(contextual information)를 활용하기 시작했으나, 여전히 3D 문맥 인식 능력의 한계로 인해 정확도를 더욱 향상시켜야 하는 과제가 남아 있다. 특히, 인접한 슬라이스 간의 미세한 차이인 'Inter-slice context residual'은 종양의 표면이나 경계에서 나타나는 핵심적인 형태학적 정보를 포함하고 있음에도 불구하고, 그 값이 매우 작아 기존 모델들은 이를 직접적으로 학습하거나 활용하지 못했다는 문제가 있다.

본 논문의 목표는 인접 슬라이스 간의 문맥 잔차(context residual)를 명시적으로 학습하고 이를 분할 성능 향상에 활용하는 3D Context Residual Network(ConResNet)를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 분할 마스크(segmentation mask)와 더불어 인접 슬라이스 간의 차이를 나타내는 잔차 마스크(residual mask)를 동시에 예측하는 듀얼 디코더 구조를 설계한 것이다.

중심적인 설계 직관은 다음과 같다. 첫째, 인접 슬라이스 간의 특징 맵(feature map) 차이를 계산하여 Inter-slice context residual을 명시적으로 추출한다. 둘째, 이렇게 추출된 잔차 정보를 다시 분할 경로의 Attention 가이드로 사용하여, 모델이 3D 공간상의 변화가 심한 영역(경계 지역)에 더 집중하게 만든다. 이를 통해 3D 문맥 인식 능력을 강화하고 최종적인 분할 정확도를 높인다.

## 📎 Related Works

논문에서는 문맥 학습(Context learning), 잔차 학습(Residual learning), 그리고 의료 영상 분할(Medical image segmentation) 세 가지 관점에서 관련 연구를 설명한다.

1. **Context learning**: 다중 스케일 정보 수집을 위한 Pyramid spatial pooling, 수용 영역(receptive field) 확장을 위한 Atrous convolution, 불필요한 정보를 필터링하는 Attention mechanism 등이 제안되었다. 하지만 이러한 방식들은 3D 의료 영상에서 특히 중요한 '슬라이스 간 잔차 정보'를 직접적으로 특성화하지 못한다는 한계가 있다.
2. **Residual learning**: 입력과 출력 사이의 잔차를 학습함으로써 망의 깊이를 깊게 쌓을 수 있게 한 He et al.의 연구가 대표적이다. 본 논문은 이 개념을 확장하여 두 인접 슬라이스 간의 특징 잔차를 인식하는 데 적용한다.
3. **Medical image segmentation**: 2D 기반의 UNet이나 3D FCN 등이 사용되어 왔다. 2D 모델은 슬라이스별로 처리하므로 슬라이스 간 문맥 정보를 잃어버리는 단점이 있으며, 기존 3D 모델들은 문맥 정보를 활용하려 하지만 본 논문에서 제안하는 명시적인 잔차 학습 방식과는 차이가 있다.

## 🛠️ Methodology

### 전체 시스템 구조

ConResNet은 Shared Encoder, Segmentation Decoder, 그리고 Context Residual (ConRes) Decoder로 구성된 Encoder-Decoder 아키텍처를 가진다. Encoder에서 추출된 특징은 두 디코더로 전달되며, 두 디코더 사이에는 각 스케일마다 Context Residual Module이 배치되어 정보를 상호 교환한다.

### 주요 구성 요소 및 역할

1. **Shared Encoder**: 9개의 Residual Block으로 구성되며, 각 블록은 두 개의 $3 \times 3 \times 3$ 합성곱 층과 Skip connection을 가진다. 배치 사이즈가 작은 환경에서도 안정적인 학습을 위해 Group Normalization과 Weight Standardization을 적용하였다. 마지막 단계에서는 수용 영역을 넓히기 위해 Dilated rate가 2인 Atrous convolution을 사용한다.
2. **Dual Decoders**:
    * **Segmentation Decoder**: 입력 영상 $X$로부터 최종 분할 마스크 $P^{seg}$를 생성한다.
    * **ConRes Decoder**: 인접 슬라이스 간의 잔차 마스크 $P^{res}$를 예측한다. 이 디코더는 Segmentation Decoder로부터 전달받은 잔차 특징을 정제하여 최종 잔차 마스크를 생성하고, 다시 Segmentation Decoder에 Attention 가이드를 제공한다.

### Context Residual Module (CRM)

CRM은 본 논문의 핵심 모듈로, 두 개의 경로(Segmentation path, Context residual path)로 구성된다.

* **Context Residual Mapping**: Segmentation 경로의 특징 $I^{seg}$와 Encoder의 Skip-connection 특징 $I^{skip}$을 합산한 후 가중치 층(Weighted layer)을 통과시켜 특징 맵 $F$를 생성한다. 이후 인접한 슬라이스 간의 절대 차이를 계산하여 잔차 특징 $G$를 구한다.
    $$G_{s+1,h,w} = |\sigma(F_{s+1,h,w}) - \sigma(F_{s,h,w})|$$
    여기서 $\sigma(\cdot)$는 Sigmoid 함수이다. 생성된 $G$는 이전 층의 잔차 특징 $I^{res}$와 결합되어 ConRes 경로의 출력 $O^{res}$가 된다.

* **Context Attention Mapping**: ConRes 경로에서 생성된 $O^{res}$를 Attention 가중치로 활용하여 Segmentation 경로의 특징을 보강한다.
    $$O^{seg} = F \otimes (1 + \sigma(O^{res}))$$
    여기서 $\otimes$는 element-wise multiplication을 의미한다. 이는 잔차가 크게 발생하는 영역(경계 지역)을 활성화하여 모델의 민감도를 높이는 역할을 한다.

### 학습 절차 및 손실 함수

모델은 분할 마스크와 잔차 마스크를 동시에 최적화하는 end-to-end 방식으로 학습된다.

1. **Segmentation Loss ($L^{seg}$)**: Binary Cross Entropy (BCE) 손실과 Dice loss의 합으로 정의된다.
    $$L^{seg} = \sum l_{bce}(P^{seg}, Y^{seg}) - \frac{2 \sum P^{seg} Y^{seg}}{\sum (P^{seg} + Y^{seg}) + \epsilon}$$
2. **Context Residual Loss ($L^{res}$)**: 잔차 마스크의 전경-배경 불균형을 해결하기 위해 클래스 가중치 $w_k$가 적용된 Weighted BCE 손실을 사용한다. 또한 학습 가속화를 위해 Deep supervision 기법을 적용하여 여러 층에서 손실을 계산한다.
    $$L^{res} = L_{res}^0 + \lambda(L_{res}^1 + L_{res}^2)$$
최종 손실 함수는 $L = L^{seg} + L^{res}$가 된다.

## 📊 Results

### 실험 설정

* **데이터셋**: BraTS 2018 (뇌종양 분할) 및 NIH Pancreas-CT (췌장 분할) 데이터셋을 사용하였다.
* **평가 지표**: Dice coefficient(겹침 정도 측정)와 Hausdorff distance(경계면 사이의 최대 거리 측정)를 사용하였다.

### 정량적 결과

1. **BraTS 데이터셋**: 6개의 최신 방법론과 비교한 결과, ConResNet은 ET(Enhancing Tumor)에서 가장 높은 Dice score를, WT(Whole Tumor)에서 가장 낮은 Hausdorff distance를 기록하였다. 특히 모든 지표의 순위를 합산한 'Sum-Score'에서 가장 낮은 수치를 기록하여 종합적인 성능이 가장 뛰어남을 입증하였다.
2. **Pancreas-CT 데이터셋**: Baseline 모델 및 7개의 최신 모델과 비교했을 때, Mean Dice(86.06%), Max Dice(92.00%), Min Dice(73.40%) 모두에서 가장 높은 성능을 달성하였다.

### 정성적 결과 및 분석

* Baseline 모델과 비교했을 때, ConResNet이 생성한 분할 결과가 실제 Ground Truth와 훨씬 유사하며, 특히 장기의 경계 부분이 더 정교하게 예측됨을 확인하였다.
* 시각화 결과, 학습된 Context residual attention map이 뇌종양의 경계 부분을 정확히 하이라이트하고 있으며, 이를 통해 특징 맵이 타겟 영역에 더 집중되는 효과가 있음을 확인하였다.

## 🧠 Insights & Discussion

### 강점 및 분석

본 연구는 3D 의료 영상에서 단순히 3D Convolution을 사용하는 것을 넘어, 인접 슬라이스 간의 '변화량(Residual)'이라는 명시적인 기하학적 정보를 학습 과정에 도입했다는 점이 매우 강력하다. Ablation study를 통해 Context residual mapping만 적용했을 때보다 Context attention mapping까지 결합했을 때 성능이 추가로 향상됨을 보였는데, 이는 잔차 정보를 추출하는 것뿐만 아니라 이를 다시 Attention으로 피드백하는 구조가 유효함을 시사한다.

### 한계 및 논의사항

1. **계산 복잡도**: 듀얼 디코더 구조로 인해 Baseline 대비 파라미터 수는 약 2.66%, 연산량(GFLOPs)은 약 18.64% 증가하였다. 다만, 이는 실시간 처리가 가능한 수준이며 성능 향상 폭에 비해 수용 가능한 수준으로 판단된다.
2. **학습 모드**: Binary mode가 Multi-class mode보다 성능이 좋게 나타났는데, 이는 각 하위 영역(ET, WT, TC)을 직접적으로 최적화하는 것이 더 효과적이기 때문으로 분석된다.
3. **확장성**: 본 제안 방식은 Axial view뿐만 아니라 Sagittal, Coronal view 등 다른 단면 방향으로도 동일하게 적용 가능하며, 실제로 유사한 성능 향상이 있음을 확인하였다. 또한 Boundary loss와 결합했을 때 상호 보완적인 효과가 있어 추가적인 성능 향상이 가능함을 보였다.

## 📌 TL;DR

본 논문은 3D 의료 영상 분할 시 인접 슬라이스 간의 미세한 형태적 차이를 학습하기 위한 **ConResNet**을 제안한다. 이 모델은 **분할 디코더와 잔차 디코더의 듀얼 구조**를 가지며, 슬라이스 간 잔차 정보를 추출하는 **Context Residual Mapping**과 이를 다시 분할 경로의 가이드로 사용하는 **Context Attention Mapping**을 통해 3D 문맥 인식 능력을 극대화한다. BraTS 및 Pancreas-CT 데이터셋에서 SOTA 성능을 달성하였으며, 이는 명시적인 3D 문맥 잔차 학습이 의료 영상의 정교한 경계 분할에 매우 효과적임을 보여준다. 향후 이 기술은 반지도 학습(semi-supervised learning)이나 예측된 잔차 마스크를 사전 정보(prior)로 활용하는 정제 네트워크로 확장될 가능성이 크다.
