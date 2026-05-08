# IMPROVING SPATIAL CODIFICATION IN SEMANTIC SEGMENTATION

Carles Ventura, Xavier Giró, Verónica Vilaplana, Kevin McGuinness, Ferran Marqués, Noel E. O’Connor (2015)

## 🧩 Problem to Solve

본 논문은 Semantic Segmentation 문제에서 지역 기술자(local descriptors)를 풀링(pooling)할 때 사용하는 공간적 부호화(spatial codification) 방식을 개선하고자 한다.

전통적인 접근 방식에서는 객체 영역(Figure)과 배경 영역(Ground)으로 나누어 특징을 추출하는 Figure-Ground (F-G) 풀링을 주로 사용해 왔다. 그러나 이러한 단순한 이분법적 구분은 객체의 경계 부분에서 발생하는 모호함과 배경 문맥(context)이 객체 묘사에 주는 부정적인 영향, 혹은 그 반대의 경우를 충분히 제어하지 못한다는 문제가 있다.

따라서 본 연구의 목표는 객체 내부와 외부 사이의 중간 지대를 도입하여 문맥의 영향을 최소화하고, 객체 내부의 공간적 구조를 더 세밀하게 표현함으로써 객체 인식 및 분할 성능을 향상시키는 것이다. 특히, 대규모 학습 데이터가 필요한 Convolutional Neural Networks (CNN)와 달리, 수작업으로 설계된 특징(manually designed features) 기반의 시스템에서도 효율적인 공간 부호화만으로 성능을 높일 수 있음을 증명하고자 한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 공간적 부호화를 정교화하기 위해 다음의 두 가지 설계를 제안하는 것이다.

첫째, 이미지 영역을 **Figure, Border, Ground**의 세 영역으로 분할하는 **F-B-G 풀링**을 제안한다. 객체 윤곽선 주변에 'Border'라는 중간 영역을 도입함으로써, 객체 묘사에 배경 문맥이 섞이는 것을 방지하는 동시에 객체 바로 인접한 영역의 풍부한 문맥 정보를 효과적으로 캡처한다.

둘째, 객체 내부(Figure region)의 묘사를 더욱 풍부하게 하기 위해 **Contour-based Spatial Pyramid (SP)**를 도입한다. 기존의 격자 기반 SP와 달리, 객체의 형상을 고려한 두 가지 구성(Crown-based SP, Cartesian-based SP)을 통해 객체 내부의 공간적 배치를 세밀하게 부호화한다.

## 📎 Related Works

본 연구는 다음과 같은 기존 연구들의 한계를 극복하고자 한다.

1. **Uijlings et al. [11]의 연구**: Figure-Border-Ground 분할의 효과를 분석했으나, 이는 Ground Truth (GT) 마스크를 사용한 이상적인 상황에서의 실험이었다. 본 논문은 이를 실제 자동 추출된 객체 후보(object candidates) 상황으로 확장하여 실용성을 검증한다.
2. **Spatial Pyramid (SP) [18]**: 이미지를 단순한 격자(grid)로 나누어 풀링하는 방식이다. 이는 이미지 전체 수준에서는 유용하지만, 객체의 실제 윤곽이나 형태를 반영하지 못한다는 한계가 있다.
3. **CNN 기반 접근 방식**: 최근 높은 성능을 보이고 있으나, ImageNet과 같은 거대한 데이터셋을 통한 사전 학습이 필수적이며 계산 비용이 매우 높다. 본 논문은 적은 데이터로도 작동하는 O2P(Second Order Pooling) 기반의 아키텍처를 개선하는 방향을 택한다.

## 🛠️ Methodology

### 1. F-B-G Spatial Pooling

본 논문은 이미지를 다음과 같이 세 가지 영역으로 정의한다.

- **Figure**: 객체의 내부 영역.
- **Border**: 객체 윤곽선 주변의 5-pixel 너비를 가진 크라운(crown) 형태의 영역.
- **Ground**: Border를 제외한 나머지 배경 영역.

이때, 각 영역의 풀링은 완전히 독립적인 격리 상태가 아니라, 지역 기술자가 계산되는 공간적 지지 범위(spatial support)를 허용하여 인접 영역의 정보가 일부 포함될 수 있도록 설계하였다. 이는 객체 인식에서 주변 문맥의 중요성을 반영하기 위함이며, 문맥에 의한 혼동을 줄이기 위해 Masked SIFT (MSIFT)와 같은 기술자를 함께 사용하여 학습 과정에서 모델이 스스로 문맥의 중요도를 조절하게 한다.

### 2. Contour-based Spatial Pyramid (SP)

Figure 영역 내부의 공간 정보를 강화하기 위해 두 가지 SP 구성을 제안한다.

- **Crown-based SP**: Figure 마스크에 거리 변환(distance transform)을 적용한 후, 로그 스케일(logarithmic base)을 기준으로 영역을 여러 층의 크라운 형태로 나눈다.
- **Cartesian-based SP**: 영역의 질량 중심(center of mass)을 원점으로 하여 Figure 영역을 4개의 기하학적 사분면(quadrants)으로 나눈다.

### 3. 훈련 및 추론 절차

전체 시스템은 다음과 같은 흐름으로 작동한다.

1. CPMC 또는 MCG 알고리즘을 통해 객체 후보군을 추출한다.
2. 각 후보군에 대해 F-B-G 영역 분할을 수행한다.
3. Figure 영역에 Cartesian-based 또는 Crown-based SP를 적용한다.
4. eSIFT, eMSIFT, eLBP 등의 지역 기술자를 추출하고, 이를 O2P(Second Order Pooling) 방식을 통해 풀링하여 최종 벡터를 생성한다.
5. 생성된 벡터를 통해 객체의 클래스를 분류하고 최종적으로 Semantic Segmentation을 수행한다.

평가 지표로는 **Average Accuracy per Category (AAC)**를 사용하며, 이는 다음과 같이 정의된다.

$$AAC = \frac{\text{Intersection}(\text{Predicted}, \text{Ground Truth})}{\text{Union}(\text{Predicted}, \text{Ground Truth})}$$

## 📊 Results

### 1. 실험 설정

- **데이터셋**: Pascal VOC 2011 및 2012.
- **객체 후보 추출**: CPMC 및 MCG.
- **비교 대상**: 기존의 Figure-Ground (F-G) 풀링 기반 O2P 아키텍처.

### 2. 주요 결과

- **이상적인 상황 (GT 마스크 사용)**:
  - F-B-G 풀링을 적용했을 때, eSIFT와 eMSIFT 기술자 모두에서 F-G 풀링보다 높은 정확도를 보였다. 특히 eMSIFT의 경우 F-B 영역만 사용하는 것이 F-G보다 성능이 좋았는데, 이는 객체 인식에 가장 중요한 문맥 정보가 객체 바로 인접 영역(Border)에 있음을 시사한다.
  - Cartesian-based SP를 Figure 영역에 적용하고 F-B-G 풀링을 결합했을 때, AAC가 최대 $75.86\%$까지 향상되었다.

- **실제 상황 (CPMC 후보군 사용)**:
  - F-B-G 풀링을 도입함으로써 VOC 2011에서 기존 $37.15$였던 성능이 $38.91$로 향상되었다.
  - 특히 `comp5` 챌린지(외부 데이터 사용 불가) 조건에서 VOC 2011은 $5.0$ 포인트, VOC 2012는 $2.3$ 포인트의 성능 이득을 얻었다.
  - Cartesian-based SP를 추가로 적용했을 때, 검증 셋(validation set)에서는 성능이 향상되었으나, 테스트 셋(test set)에서는 일부 하락하는 경향이 관찰되었다.

- **강건성 검증 (MCG 후보군 사용)**:
  - MCG 기반 후보군을 사용할 때도 F-G 풀링($30.88$)보다 F-B-G 풀링($34.09$)의 성능이 월등히 높았으며, 여기에 Cartesian-based SP까지 적용하면 $36.10$까지 성능이 올라갔다. 이는 제안 방법이 특정 후보군 추출 알고리즘에 종속되지 않고 강건하게 작동함을 보여준다.

## 🧠 Insights & Discussion

본 논문은 공간적 부호화의 정교함이 Semantic Segmentation 성능에 직접적인 영향을 미친다는 것을 입증하였다. 특히 **Border 영역의 도입**은 배경의 노이즈를 줄이면서도 핵심적인 문맥 정보를 보존하는 매우 효율적인 장치임을 확인하였다.

또한, 객체 내부의 공간 구조를 반영하는 **Contour-based SP**가 유의미한 성능 향상을 가져온다는 점을 발견하였다. 다만, SP 적용 시 검증 셋과 테스트 셋의 결과가 상이하게 나타난 점은 주목할 만하다. 이는 SP가 특정 데이터셋의 공간적 특징에 과적합(overfitting)되었을 가능성이 있으며, 객체의 기하학적 다양성에 따라 SP의 효과가 달라질 수 있음을 시사한다.

결론적으로, 대규모 데이터셋 없이도 특징 추출 단계에서의 공간적 전략 수립만으로 CNN 기반 모델이 아닌 전통적인 풀링 기반 모델의 한계를 어느 정도 극복할 수 있음을 보여준 연구이다.

## 📌 TL;DR

본 논문은 Semantic Segmentation의 성능 향상을 위해 이미지 영역을 **Figure, Border, Ground**로 세분화하는 **F-B-G 풀링**과 객체 내부 구조를 정밀하게 묘사하는 **Contour-based Spatial Pyramid**를 제안하였다. 실험 결과, 제안된 방식은 Pascal VOC 데이터셋에서 기존 Figure-Ground 방식보다 우수한 성능을 보였으며, 특히 객체 주변의 Border 영역이 인식 성능에 결정적인 역할을 함을 밝혔다. 이 연구는 데이터 효율적인 객체 인식 및 분할 시스템 구축을 위한 공간 부호화의 중요성을 제시하였다.
