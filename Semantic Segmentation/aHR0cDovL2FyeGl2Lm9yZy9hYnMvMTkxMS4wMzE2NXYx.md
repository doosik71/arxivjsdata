# Gated Graph Convolutional Neural Network와 Deep Structured Feature Embedding을 통한 건물 분할
Yilei Shi, Qingyu Li, Xiao Xiang Zhu

## 🧩 Problem to Solve
광학 이미지에서 건물을 자동 추출하는 것은 건물의 복잡한 모양 때문에 여전히 어려운 문제입니다. 심층 합성곱 신경망(DCNN) 기반의 시맨틱 분할은 픽셀 단위 분류에서 뛰어난 성능을 보였지만, 점진적인 다운샘플링으로 인해 경계면을 정밀하게 구분하는 데 어려움을 겪고, 결과적으로 경계가 불분명하거나 뭉툭한 형태의 분할 결과를 초래합니다. 또한, DCNN은 픽셀 간의 상호작용을 충분히 고려하지 않아 미세한 지역적 세부 정보를 놓치는 경향이 있습니다. 기존의 DCNN과 조건부 확률장(CRF)을 결합한 방법들도 특징 추출의 부족과 정보 전파의 한계가 있었습니다.

## ✨ Key Contributions
*   **DSFE-GGCN 프레임워크 제안**: 딥 스트럭처드 특징 임베딩(Deep Structured Feature Embedding, DSFE)과 그래프 합성곱 신경망(Graph Convolutional Network, GCN)을 통합하는 포괄적인 시맨틱 분할 엔드-투-엔드 프레임워크를 제안합니다.
*   **게이팅된 그래프 합성곱 신경망(GGCN) 개발**: 장거리 정보 전파를 위해 GRU(Gated Recurrent Unit)를 사용하는 순환 신경망(RNN)과 단거리 정보 전파를 위한 GCN을 결합한 새로운 네트워크 아키텍처인 GGCN을 제안합니다.
*   **효과적인 데이터 전처리 기법 제안**: 중해상도 위성 이미지를 위한 4단계(밴드 정규화, 공동 등록, 개선, Truncated Signed Distance Map, TSDM) 전처리 및 데이터 증강 방법을 제안합니다.
*   **최첨단 성능 달성**: 제안하는 GGCN이 통합된 프레임워크가 건물 외곽선 추출에서 기존의 최첨단 접근 방식들을 능가하는 성능을 보임을 체계적인 실험을 통해 입증합니다.

## 📎 Related Works
*   **DCNN 기반 시맨틱 분할**: FCN (Fully Convolutional Network)은 완전 연결 계층을 합성곱 계층으로 대체하여 효율성을 높였습니다. SegNet은 맥스 풀링 인덱스를 재사용하여 메모리 효율성을 개선했고, U-Net은 긴 스킵 연결을 통해 손실된 공간 정보를 복구합니다. DeepLab-CRF는 CRF를 DCNN의 마지막 계층에 통합하여 엔드-투-엔드 학습을 가능하게 했습니다.
*   **그래프 모델**: Markov Random Field (MRF) 및 Conditional Random Field (CRF)와 같은 확률적 그래프 모델은 픽셀 간의 관계를 고려하여 정밀한 경계를 생성하는 데 사용되었습니다. 최근에는 그래프 합성곱 신경망(GCN)이 그리드 구조가 아닌 데이터에 대한 딥러닝을 가능하게 하며, 계산 효율성을 개선했습니다.
*   **건물 외곽선 추출**: 다단계 ConvNet, 능동 윤곽 모델(Active Contour Model, ACM), 잔차 개선 네트워크(Residual Refinement Network), 조건부 Wasserstein 생성적 적대 신경망(CWGAN-GP) 등 다양한 딥러닝 기반 방법들이 제안되었습니다.

## 🛠️ Methodology
제안하는 DSFE-GGCN 프레임워크는 이미지를 노드가 픽셀인 그래프로 일반화하여 처리합니다.

1.  **Deep Structured Feature Embedding (DSFE)**:
    *   DCNN을 특징 추출기로 사용하여 이미지로부터 다양한 레벨의 특징 벡터를 추출합니다.
    *   낮은 레벨의 특징은 공간 세부 정보가 풍부하고, 높은 레벨의 특징은 포괄적인 시맨틱 정보를 제공하므로, 이들을 점진적으로 연결(concatenate)하여 공간 정보, 시맨틱 정보 및 기타 속성을 그래프 합성곱 신경망으로 전파합니다.
    *   실험을 통해 FC-DenseNet이 가장 효과적인 특징 추출기로 선택되었습니다.

2.  **Gated Graph Convolutional Neural Network (GGCN)**:
    *   **전파 모델(Propagation Model)**:
        *   메시지 계층 ($a_{t}^{i}$)은 이웃 노드($V_{i}$)로부터 정보를 집계합니다. GCN이 메시지 함수로 사용되어 노드 $i$에서 도달 가능한 모든 노드로 임베딩이 전파되도록 합니다. GCN 계층은 다음과 같이 정의됩니다:
            $$a_{t}^{i} = \sigma_{r}(\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}Wh_{t-1}^{i})$$
            여기서 $\tilde{A}$는 자체 연결이 추가된 인접 행렬, $\tilde{D}$는 해당 차수 행렬 ($\tilde{D}_{ii} = \sum_{j} \tilde{A}_{ij}$), $W$는 학습 가능한 가중치 행렬, $\sigma_{r}$은 ReLU 함수입니다.
        *   숨겨진 계층 ($h_{t}^{i}$)은 게이팅된 순환 유닛(GRU)을 사용하여 이전 숨겨진 상태($h_{t-1}^{i}$)와 메시지 계층($a_{t}^{i}$)을 기반으로 업데이트됩니다. GRU는 재설정 게이트($r_{t}^{i}$)와 업데이트 게이트($z_{t}^{i}$)를 사용하여 노드가 자체 메모리를 유지하고 유용한 정보를 추출하며 장거리 의존성을 포착할 수 있도록 합니다:
            $$r_{t}^{i} = \sigma_{s}(W_{r}h_{t-1}^{i} + U_{r}a_{t}^{i})$$
            $$z_{t}^{i} = \sigma_{s}(W_{z}h_{t-1}^{i} + U_{z}a_{t}^{i})$$
            $$\tilde{h}_{t}^{i} = \tanh(W(r_{t}^{i}\circ h_{t-1}^{i}) + Ua_{t}^{i})$$
            $$h_{t}^{i} = (1-z_{t}^{i})\circ h_{t-1}^{i} + z_{t}^{i}\circ \tilde{h}_{t}^{i}$$
            여기서 $\sigma_{s}$는 시그모이드 함수, $\circ$는 내적을 나타냅니다.
    *   **예측 모델(Prediction Model)**: 최종 시간 단계 $t+n$의 숨겨진 상태 $h_{t+n}^{i}$에 소프트맥스 함수를 적용하여 각 노드의 클래스 확률을 예측합니다:
        $$p = \text{softmax}(h_{t+n}^{i})$$
    *   손실 함수로는 음의 로그 우도(Negative Log-Likelihood, NLL-Loss)가 사용됩니다.

3.  **전처리(Preprocessing)**:
    *   **밴드 정규화(Band Normalization)**: 이미지 밴드를 정규화합니다.
    *   **공동 등록(Coregistration)**: OSM 건물 외곽선과 위성 이미지 간의 불일치를 보정합니다. 가우시안 기울기의 교차 상관을 통해 오프셋을 계산합니다.
    *   **개선(Refinement)**: 추가적인 개선 작업을 수행합니다.
    *   **Truncated Signed Distance Map (TSDM)**: 픽셀을 건물 경계까지의 거리로 표현하여 멀티-라벨 분할 문제로 변환합니다. 경계 내부 픽셀은 양수, 외부 픽셀은 음수 값을 가지며, 특정 임계값 ($T_{d}$)에서 거리가 절단됩니다.
        $$D(x) = \delta_{d} \cdot \min(\min_{x \in X}(d(x)), T_{d})$$
        여기서 $d(x)$는 유클리드 거리, $\delta_{d}$는 부호 함수입니다.

## 📊 Results
*   **데이터셋**: Planetscope 위성 이미지 (3m 해상도) 및 ISPRS 포츠담 데이터셋 (5cm 해상도 항공 이미지).
*   **평가 지표**: 전체 정확도(OA), F1 점수, IoU(Intersection over Union) 점수.
*   **DCNN 베이스라인**: FC-DenseNet이 DCNN 중 가장 우수한 성능을 보였으며 (Planetscope: IoU 0.4628), FCN-8s 및 U-Net이 낮은 수준 특징 연결 덕분에 다른 FCN보다 나은 결과를 보였습니다.
*   **DSFE를 사용한 GCN**: FC-DenseNet을 특징 추출기로 사용했을 때 DSFE(FC-DenseNet)-GCN이 가장 좋은 성능을 보였습니다 (Planetscope: IoU 0.5012). 이는 FC-DenseNet의 다중 레벨 특징 추출 능력이 우수함을 입증합니다.
*   **다양한 그래프 모델과의 비교**: 제안된 DSFE-GGCN 프레임워크가 모든 비교 대상 모델 중에서 가장 우수한 성능을 달성했습니다.
    *   **Planetscope 데이터셋**: DSFE-GGCN은 IoU 0.5251 (최고 DCNN 대비 6.2% 증가)를 기록했습니다.
    *   **ISPRS 포츠담 데이터셋**: DSFE-GGCN은 IoU 0.9196 (DSFE-GCN 대비 0.99% 증가)을 달성하여, 고해상도 이미지에서도 탁월한 성능을 보였습니다.
*   **시각적 결과**: DSFE-GGCN은 다른 방법에 비해 더 선명한 경계와 더 완전한 건물 추출 결과를 보여주었습니다.

## 🧠 Insights & Discussion
*   이 연구의 핵심 통찰은 DCNN의 특징 추출 능력(DSFE)과 그래프 모델의 픽셀 간 상호작용 및 공간 정보 전파 능력(GGCN)을 결합할 때 시맨틱 분할, 특히 정밀한 경계 추출에서 상당한 성능 향상을 얻을 수 있다는 점입니다.
*   GGCN의 GRU 기반 게이팅 메커니즘은 바닐라 GCN이 놓치기 쉬운 장거리 의존성을 포착하여 모델이 더욱 포괄적인 문맥 정보를 활용하게 합니다.
*   TSDM과 같은 전처리 단계는 네트워크가 클래스 라벨과 함께 기하학적 속성을 학습하도록 돕고, 이는 예측 결과의 정확도를 높이는 데 기여합니다.
*   제안된 프레임워크는 건물 외곽선 추출 외에도 도로 추출, 정착지 계층 추출, 또는 일반적인 고해상도 데이터의 시맨틱 분할과 같은 다른 이진 또는 멀티-라벨 분할 작업에 일반화하여 적용될 수 있습니다.
*   또한, GGCN은 포인트 클라우드나 소셜 미디어 메시지와 같은 비정형 데이터에도 직접 적용될 수 있는 잠재력을 가지고 있습니다.

## 📌 TL;DR
**문제**: 기존 DCNN은 다운샘플링으로 인해 시맨틱 분할 시 경계가 불분명하고 미세한 지역적 세부 정보를 놓치는 문제가 있었습니다.

**방법**: 본 논문은 딥 스트럭처드 특징 임베딩(DSFE)과 새로운 게이팅된 그래프 합성곱 신경망(GGCN)을 결합한 엔드-투-엔드 프레임워크를 제안합니다. DSFE는 다중 레벨 특징을 추출하고, GGCN은 GCN을 통한 단거리 정보 전파와 GRU를 통한 장거리 정보 전파를 결합하여 픽셀 간의 문맥적 관계를 효과적으로 모델링합니다.

**핵심 결과**: 제안된 DSFE-GGCN은 건물 외곽선 추출 작업에서 기존의 최첨단 DCNN 및 다른 그래프 모델들을 뛰어넘는 성능을 보였습니다. 특히, 경계면을 더 선명하게 구분하고 픽셀 단위 분류의 정확도를 크게 향상시켜, 중해상도 위성 이미지와 고해상도 항공 이미지 모두에서 우수함을 입증했습니다.