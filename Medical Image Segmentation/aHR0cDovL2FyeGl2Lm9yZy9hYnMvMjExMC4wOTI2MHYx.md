# A Unified Framework for Generalized Low-Shot Medical Image Segmentation with Scarce Data

Hengji Cui, Dong Wei, Kai Ma, Shi Gu, and Yefeng Zheng (2021)

## 🧩 Problem to Solve

본 논문은 의료 영상 분할(Medical Image Segmentation) 분야에서 발생하는 데이터 및 어노테이션의 극심한 부족 문제를 해결하고자 한다. 일반적으로 딥러닝 기반의 분할 네트워크는 학습을 위해 방대한 양의 데이터와 정교한 라벨링을 필요로 하지만, 의료 영상의 경우 전문가의 수작업이 필수적이므로 비용과 시간이 많이 소요된다. 특히 희귀 질환(Rare diseases)의 경우에는 어노테이션뿐만 아니라 학습에 사용할 수 있는 원본 데이터 자체의 양이 매우 제한적이다.

기존의 Low-shot learning 연구들은 주로 두 가지 가정에 의존하였다. 첫째, 자연어 이미지 분야의 Low-shot segmentation은 대량의 데이터가 있는 'Base classes'에서 사전 지식을 학습한 후 이를 'Novel classes'에 적용하는 방식을 취하지만, 의료 영상에서는 분할 대상이 이미 한정되어 있어 이러한 설정이 불가능한 경우가 많다. 둘째, 최근의 의료 영상 Low-shot 연구들은 이미지 합성(Image Synthesis)이나 준지도 학습(Semi-supervised learning)을 활용하여 부족한 라벨을 보완하려 하지만, 이는 '라벨이 없는 데이터'는 풍부하다는 가정하에 작동한다. 따라서 본 논문의 목표는 데이터와 어노테이션이 모두 극도로 부족한 상황에서도 강건하게 작동하는 일반화된 Low-shot 의료 영상 분할 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Distance Metric Learning (DML)을 기반으로 각 카테고리를 단일 프로토타입이 아닌 '다중 모드 표현(Multimodal representation)'으로 학습하는 것이다. 이를 통해 데이터 부족으로 인한 과적합(Overfitting)을 방지하고, 환자 간의 유사성(Inter-subject similarity)과 클래스 내부의 변동성(Intraclass variations)을 동시에 활용한다. 구체적인 기여 사항은 다음과 같다.

1. **MRE-Net 제안**: 데이터와 어노테이션이 모두 부족한 상황에서 작동하는 Unified framework인 Multimodal Representation Embedding Network (MRE-Net)를 제안하였다.
2. **적응형 혼합 계수(Adaptive Mixing Coefficients)**: 각 카테고리의 다중 모드 중 현재 입력 이미지에 가장 적합한 모드에 더 높은 가중치를 부여하는 self-attention 메커니즘을 도입하였다.
3. **효율적인 구현**: 카테고리 표현을 Fully Connected (FC) 레이어의 가중치로 암시적으로 임베딩하여, 코사인 유사도 계산을 순전파(Forward propagation) 과정에서 효율적으로 수행하도록 설계하였다.
4. **의료 영상 특화 전략**: 클래스 불균형 해결을 위한 Online Hard Example Mining (OHEM), 구조적 유사성 활용을 위한 Cartesian coordinates 추가, 다양한 타겟 대응을 위한 Attentional Multi-Scale (AMS) 임베딩을 통합하였다.

## 📎 Related Works

본 논문에서는 기존의 접근 방식들을 다음과 같이 분류하고 그 한계를 지적한다.

- **DNN 기반 분할 (U-Net 등)**: 현재 의료 영상 분할의 표준이지만, 학습을 위해 매우 많은 데이터를 요구하는 'Data-hungry' 특성이 있어 Low-shot 상황에서는 성능이 급격히 저하된다.
- **DML 기반 Low-shot 학습**: 자연어 이미지 분야에서 성공적이었으나, 앞서 언급한 Base/Novel class 설정의 차이로 인해 의료 영상에 직접 적용하기 어렵다. 다만, 거리 기반의 표현 학습 아이디어는 유효하다고 판단하여 이를 채택하였다.
- **이미지 합성 및 준지도 학습**: 이미지 변환이나 GAN을 이용해 데이터를 증강하는 방식은 다량의 unlabeled data가 필요하므로, 데이터 자체가 희귀한 상황에서는 적용 불가능하다.
- **등록 기반 분할 (Registration-based, e.g., ANTs)**: Atlas를 이용하는 전통적인 방식으로 Low-shot 상황에서 어느 정도 성능을 내지만, 개별 환자 간의 외형/구조적 변동성이 클 경우 효과가 떨어지며 계산 시간이 매우 오래 걸린다는 단점이 있다.

## 🛠️ Methodology

### 전체 시스템 구조
MRE-Net은 입력 이미지를 임베딩 공간으로 투영하고, 학습된 카테고리 프로토타입과의 거리를 계산하여 픽셀 단위의 분할을 수행하는 구조이다. 전체 파이프라인은 **Backbone $\rightarrow$ AMS Embedding $\rightarrow$ DML Dense Prediction** 순으로 구성된다.

### 주요 구성 요소 및 역할
1. **Backbone Network**: modified 3D U-Net을 사용하여 특징을 추출한다. 기존 U-Net의 skip connection에 transition convolution을 추가하여 저수준 특징과 고수준 특징의 채널 수를 조정함으로써 성능을 높였다.
2. **Attentional Multi-Scale (AMS) Embedding**: U-Net의 각 레벨에서 추출된 특징 맵들을 결합하고, 여기에 픽셀의 정규화된 좌표 정보(Cartesian coordinates)를 추가 채널로 입력한다. 이후 SE block(Channel-wise attention)과 ASPP(Atrous Spatial Pyramid Pooling)를 거쳐 다중 스케일의 특징을 통합한 최종 임베딩 벡터 $e_i$를 생성한다.
3. **Multimodal Representation Embedding**: 각 클래스 $k$를 $M$개의 프로토타입 $\{c_{k,j}\}_{j=1}^M$의 집합으로 정의한다. 이를 통해 단일 벡터로는 표현하기 힘든 클래스 내부의 다양한 변동성을 캡처한다.

### 학습 목표 및 방정식
#### 1. 거리 기반 확률 계산
픽셀 임베딩 $e_i$와 프로토타입 $c_{k,j}$ 사이의 거리를 $d(e_i, c_{k,j})$라고 할 때, 픽셀 $i$가 클래스 $k$에 속할 확률 $P(s_i=k)$는 다음과 같이 정의된다.
$$P(s_i=k) = \frac{\sum_{j=1}^{M} \alpha_{k,j} \exp(\xi \hat{e}_i^T \hat{c}_{k,j})}{\sum_{k=1}^{K} \sum_{j=1}^{M} \alpha_{k,j} \exp(\xi \hat{e}_i^T \hat{c}_{k,j})}$$
여기서 $\hat{e}_i$와 $\hat{c}_{k,j}$는 $L_2$ 정규화된 벡터이며, $\xi$는 학습 가능한 스케일링 인자이다.

#### 2. 적응형 혼합 계수 ($\alpha_{k,j}$)
모든 모드에 동일한 가중치를 주는 대신, 입력 $e_i$에 따라 가중치를 조절하는 self-attention 메커니즘을 사용한다.
$$\alpha_{k,j} = \frac{\exp(\beta_{k,j})}{\sum_{m=1}^{M} \exp(\beta_{k,m})}$$
여기서 $\beta$는 $e_i$를 입력으로 받는 두 개의 FC 레이어(Squeeze & Excitation 구조)를 통해 계산된다.

#### 3. 손실 함수 및 학습 절차
손실 함수로는 클래스별로 정규화된 Cross-Entropy Loss를 사용한다.
$$L_{CE} = -\sum_{k} \frac{\delta_k(a_i) \log[P(s_i=k)]}{N_k}$$
또한, 클래스 불균형 문제를 해결하기 위해 **OHEM (Online Hard Example Mining)**을 적용하여, 다수 클래스(Majority group)에서는 손실 값이 큰 어려운 샘플들만 선택적으로 학습에 참여시킨다.

## 📊 Results

### 실험 설정
- **데이터셋**: MRBrainS18 (뇌 MRI, 8개 구조 분할), BTCV (복부 CT, 8개 장기 분할).
- **비교 대상**: 3D U-Net, ANTs (Registration-based), sSE method, DataAug (Semi-supervised).
- **지표**: Dice Similarity Coefficient (Dice), 95th percentile Hausdorff Distance ($HD_{95}$).
- **설정**: One-shot ($L=1$) 및 Few-shot ($L=2, 3$) 시나리오에서 실험을 수행하였다.

### 주요 결과
1. **One-shot 성능**: 
   - MRBrainS18 데이터셋에서 MRE-Net-1은 평균 Dice $78.39\%$, $HD_{95}$ $6.30\text{mm}$를 기록하여, U-Net-1 ($14.86\%$)과 ANTs ($65.93\%$)를 압도하였다.
   - BTCV 데이터셋에서도 MRE-Net-1은 평균 Dice $69.13\%$, $HD_{95}$ $19.02\text{mm}$를 달성하여, 다른 모든 방법론보다 월등한 성능을 보였다. 특히 ANTs와 U-Net은 복부 CT의 큰 개체 간 변동성으로 인해 유의미한 결과를 내지 못했다.
2. **Few-shot 확장성**: 학습 샘플 수가 1개에서 3개로 증가함에 따라 성능이 점진적으로 향상되었으며, MRE-Net-3는 U-Net-6(6개 샘플 학습)보다 더 나은 성능을 보이기도 하였다.
3. **시간 효율성**: 추론 시간은 U-Net과 거의 동일(약 4초)한 반면, ANTs는 샘플당 약 28분이 소요되어 실시간 적용 가능성 면에서 큰 우위를 점하였다.
4. **Ablation Study**: DML의 도입이 성능 향상에 가장 결정적인 역할을 하였으며, 그 다음으로 Cartesian coordinates와 AMS embedding, OHEM 순으로 기여도가 높음을 확인하였다.

## 🧠 Insights & Discussion

### 강점 및 해석
본 연구의 가장 큰 강점은 **DML을 통한 비파라메트릭(Non-parametric) 접근 방식**을 채택하여 극소량의 데이터에서도 과적합을 효과적으로 억제했다는 점이다. t-SNE 시각화 결과, MRE-Net은 3D U-Net에 비해 클래스별 특징 공간을 훨씬 더 명확하고 조밀하게 클러스터링하는 것으로 나타났다. 또한, 단일 프로토타입이 아닌 다중 모드 표현을 사용함으로써 동일 장기라도 환자마다 다를 수 있는 형태적/강도적 변동성을 효과적으로 모델링할 수 있었다.

### 한계 및 논의사항
- **계산 자원**: DML 특성상 모든 픽셀과 프로토타입 간의 거리 계산이 필요하므로, 일반 U-Net보다 GPU 메모리 사용량이 많고 학습 시간이 약간 더 소요된다.
- **불규칙한 구조 분할**: 뇌의 백질 고혈압(WMH)과 같이 형태와 위치가 매우 불규칙한 영역의 분할 성능은 상대적으로 낮았다. 이는 현재의 거리 기반 매칭 방식만으로는 해결하기 어려운 문제이며, 향후 특화된 전략이 필요함을 시사한다.
- **전이 학습의 미적용**: 본 논문에서는 scratch부터 학습하였으나, 다른 도메인에서 사전 학습된 모델을 전이 학습(Transfer Learning)시킨다면 데이터 부족 문제를 더욱 완화할 수 있을 것으로 보인다.

## 📌 TL;DR

본 논문은 데이터와 어노테이션이 모두 극도로 부족한 희귀 질환 등의 의료 영상 분할 문제를 해결하기 위해 **MRE-Net**이라는 프레임워크를 제안하였다. 핵심은 **Distance Metric Learning (DML)**과 **다중 모드 프로토타입(Multimodal Prototype)**을 결합하여 환자 간 유사성과 내부 변동성을 동시에 학습하는 것이다. 실험 결과, 단 1~3개의 샘플만으로도 기존의 U-Net이나 전통적인 Registration 방식보다 훨씬 뛰어난 분할 성능과 빠른 추론 속도를 보였으며, 이는 데이터 획득이 어려운 임상 환경에서 매우 실용적인 솔루션이 될 가능성이 높다.