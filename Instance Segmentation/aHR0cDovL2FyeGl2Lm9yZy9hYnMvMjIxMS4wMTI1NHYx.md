# CircleSnake: Instance Segmentation with Circle Representation

Ethan H. Nguyen, Haichun Yang, Zuhayr Asad, Ruining Deng, Agnes B. Fogo, and Yuankai Huo (2022)

## 🧩 Problem to Solve

본 연구가 해결하고자 하는 핵심 문제는 구형(ball-shaped)의 의료 영상 객체, 특히 신장의 사구체(glomeruli)를 정밀하게 분할(instance segmentation)하는 것이다. 기존의 컴퓨터 비전 분야에서 널리 사용되는 Bounding Box나 Polygon 표현 방식은 일반적인 자연 이미지에는 적합하지만, 의료 영상의 특성상 객체가 어느 각도로든 회전되어 나타날 수 있는 환경에서는 효율성이 떨어진다.

기존의 다각형 기반 표현 방식은 회전 시 일관성이 부족하고, 불필요하게 많은 자유도(Degrees of Freedom, DoF)를 가져 학습의 불안정성을 초래할 수 있다. 따라서 본 논문의 목표는 구형 객체에 최적화된 Circle representation을 인스턴스 분할 단계까지 확장하여, 회전 일관성을 확보하고 더 적은 파라미터로 정밀한 분할 성능을 달성하는 CircleSnake 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 중심적인 아이디어는 객체 검출(detection) 단계부터 분할(segmentation) 단계까지 일관되게 원(circle) 형태의 표현법을 사용하는 것이다. 주요 기여 사항은 다음과 같다.

1.  **효율적인 윤곽선 제안(Contour Proposal):** DeepSnake와 같이 복잡한 Bounding box를 팔각형(octagon)으로 변환하는 과정 대신, 계산 비용이 거의 없는 'Bounding Circle $\rightarrow$ Circle Contour' 적응 방식을 도입하여 파이프라인을 단순화하였다.
2.  **자유도(DoF) 감소를 통한 강건성 확보:** 팔각형 표현(DoF=8)에 비해 원 표현(DoF=2, 반지름)은 훨씬 적은 자유도를 가진다. 이는 모델이 더 강건한 성능을 보이게 하며, 특히 객체의 회전과 관계없이 일관된 초기 윤곽선을 제공한다.
3.  **End-to-End 원 표현 기반 분할 파이프라인:** 원 검출(Circle Detection), 원 윤곽선 제안(Circle Contour Proposal), 원형 컨볼루션(Circular Convolution)을 통합한 최초의 end-to-end 딥러닝 분할 파이프라인을 구축하였다.

## 📎 Related Works

논문에서는 다음과 같은 기존 접근 방식과 그 한계를 언급한다.

-   **Pixel-based Methods:** Mask R-CNN과 같은 방식은 지역 제안(regional proposal) 내에서 픽셀 단위로 분할을 수행한다. 하지만 이는 구형 객체의 기하학적 특성을 충분히 활용하지 못한다.
-   **DeepSnake:** 검출기의 위치 오류를 해결하기 위해 초기 팔각형 윤곽선을 객체 경계로 변형(deformation)시키는 방식이다. 그러나 구형의 사구체를 표현하기에 팔각형은 최적이 아니며, Bounding box에서 팔각형으로 변환하는 과정이 계산적으로 복잡하다.
-   **CircleNet:** 구형 의료 객체 검출을 위해 원 표현법을 도입하여 우수한 성능을 보였다. 본 연구는 이 CircleNet의 검출 능력을 분할 단계까지 확장하여 CircleSnake를 구현하였다.

## 🛠️ Methodology

CircleSnake의 전체 파이프라인은 '원 검출 $\rightarrow$ 원 윤곽선 제안 $\rightarrow$ 윤곽선 변형'의 순서로 진행된다.

### 1. Circle Object Detection
CircleNet의 설계를 따라 객체의 중심점과 반지름을 예측한다.
-   **중심점 예측:** 히트맵 $\hat{Y}$를 생성하며, 정답 중심점은 2D 가우시안 커널로 표현된다.
    $$Y_{xyc} = \exp\left(-\frac{(x-\tilde{p}_x)^2 + (y-\tilde{p}_y)^2}{2\sigma_p^2}\right)$$
-   **손실 함수:** Focal loss를 기반으로 한 penalty-reduced logistic regression $L_k$와 위치 정밀도를 위한 $\ell_1$-norm offset loss $L_{off}$를 사용한다.
-   **반지름 예측:** 각 픽셀에 대한 반지름 $\hat{R}$을 예측하며, $\ell_1$ 손실 함수를 통해 최적화한다.
    $$L_{radius} = \frac{1}{N}\sum_{k=1}^N ||\hat{R}_{p_k} - r_k||$$
-   **전체 검출 손실 함수:**
    $$L_{det} = L_k + \lambda_{radius}L_{radius} + \lambda_{off}L_{off}$$

### 2. Circle Contour Proposal
검출된 중심점 $\hat{p}$와 반지름 $\hat{r}$을 이용하여 초기 원 윤곽선을 생성한다. 복잡한 변환 과정 없이, 원의 둘레를 따라 $N=128$개의 점을 균일하게 샘플링하여 초기 정점(vertices) $\{x^{circle}_i\}_{i=1}^N$을 구성한다. 이는 계산 비용이 거의 없으며 회전에 대해 매우 일관적인 초기값을 제공한다.

### 3. Circular Contour Deformation
초기 원 윤곽선을 실제 객체의 경계로 변형시키기 위해 Circular Convolution과 GCN(Graph Convolutional Network)을 사용한다.
-   **입력 특징:** 각 정점 $x^{circle}_i$의 특징 벡터는 학습된 특징 맵 $F(x^{circle}_i)$와 정점의 좌표 $x^{circle}_i$를 결합하여 생성한다.
-   **원형 컨볼루션(Circular Convolution):** 정점들을 원형 1차원 신호로 취급하여 주기적 컨볼루션을 수행한다.
    $$(f^{circle}_N * k)_i = \sum_{j=-r}^r (f^{circle}_N)_{i+j} k_j$$
-   **GCN 구조:** 8개의 'CirConv-BN-ReLU' 레이어로 구성된 Backbone, 다양한 스케일의 특징을 결합하는 Fusion 블록, 그리고 최종적으로 정점별 오프셋(offset)을 예측하는 Prediction head로 이루어져 있다.
-   **변형 학습:** 예측된 오프셋을 통해 정점을 이동시키며, 실제 경계점 $x_{gt,i}$와의 $\ell_1$ 거리 손실을 최소화한다.
    $$L_{iter} = \frac{1}{N}\sum_{i=1}^N \ell_1(\tilde{x}^{circle}_i - x_{gt,i})$$
    이 과정은 총 3회 반복 수행되어 최종 윤곽선을 완성한다.

## 📊 Results

### 실험 설정
-   **데이터셋:** 신장 생검(renal biopsies) 이미지에서 추출한 사구체 데이터셋을 사용하였다. (학습 7,040장, 검증 980장, 테스트 1,470장)
-   **백본 네트워크:** DLA(Deep Layer Aggregation)를 사용하였다.
-   **비교 대상:** CenterNet, CircleNet, DeepSnake.
-   **평가 지표:** 검출 성능은 Average Precision (AP)으로, 분할 성능은 Dice score로 측정하였다.

### 주요 결과
-   **검출 성능:** CircleSnake는 모든 AP 지표에서 가장 우수한 성능을 보였다. 특히 Segmentation 버전의 CircleSnake는 $AP$ 0.614를 기록하여 DeepSnake(0.559)보다 높았다.
-   **분할 성능:** 사구체 분할의 Dice score에서 CircleSnake는 **0.849**를 달성하여, DeepSnake의 **0.804** 대비 유의미한 향상을 보였다.
-   **정성적 결과:** 시각화 결과, CircleSnake가 수동 어노테이션(Manual Seg)에 더 근접한 정밀한 경계를 생성함을 확인하였다.

## 🧠 Insights & Discussion

본 논문은 의료 영상의 기하학적 특성을 고려한 표현법의 선택이 모델의 성능과 강건성에 얼마나 큰 영향을 미치는지를 잘 보여준다. 

**강점 및 해석:**
-   **DoF의 효율성:** 팔각형(DoF=8) 대신 원(DoF=2)을 사용함으로써 모델이 최적화해야 할 파라미터 공간을 줄였으며, 이는 곧 회전 일관성(Rotation Consistency)의 향상으로 이어졌다.
-   **파이프라인의 단순화:** Bounding box $\rightarrow$ Octagon으로 이어지는 복잡한 변환 과정을 단순한 원 샘플링으로 대체함으로써 end-to-end 학습의 효율성을 높였다.

**한계 및 논의사항:**
-   **객체 형태의 제약:** 본 모델은 '구형(ball-shaped)' 객체에 최적화되어 있다. 따라서 타원형이나 불규칙한 모양의 객체에 적용할 경우, 초기 원 표현이 적절한 시작점이 되지 못할 가능성이 있다.
-   **반복 횟수의 고정:** 3회의 반복 변형(iteration)을 수행하는데, 객체의 형태가 원에서 많이 벗어난 경우 더 많은 반복이 필요할 수 있으나 이에 대한 동적 조절 메커니즘은 제시되지 않았다.

## 📌 TL;DR

CircleSnake는 구형 의료 객체 분할을 위해 **원 표현(Circle Representation)**을 도입한 end-to-end 인스턴스 분할 프레임워크이다. 복잡한 다각형 제안 대신 단순한 원 기반의 초기 윤곽선을 생성하고 이를 **Circular Convolution** 기반의 GCN으로 변형시켜 정밀한 경계를 찾아낸다. 이를 통해 기존 DeepSnake 대비 낮은 자유도(DoF)로도 더 높은 회전 일관성과 분할 정확도(Dice score 0.804 $\rightarrow$ 0.849)를 달성하였으며, 이는 향후 다른 구형 의료 객체(핵, 종양 등)의 분할 연구에도 유용하게 활용될 가능성이 높다.