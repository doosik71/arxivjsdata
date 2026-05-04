# U-Net and its variants for medical image segmentation: theory and applications

Nahian Siddique, Paheding Sidike, Colin Elkin and Vijay Devabhaktuni (2020)

## 🧩 Problem to Solve

본 논문은 의료 영상 분석, 특히 이미지 세그멘테이션(Image Segmentation) 분야에서 표준적으로 사용되는 U-Net 아키텍처와 그 변형 모델들을 체계적으로 분석하는 것을 목표로 한다. 

의료 영상 분야에서 세그멘테이션은 매우 중요하지만 다음과 같은 고유한 문제점들이 존재한다. 첫째, 전문의의 정밀한 레이블링이 필요하기 때문에 학습에 사용할 수 있는 정답 데이터(Annotated Data)가 매우 희소하다. 둘째, 단순한 분류(Classification)와 달리 픽셀 수준의 문맥 정보(Pixel-level context information)가 필수적이다. 셋째, 서로 맞닿아 있는 동일 클래스의 객체들을 정확히 분리해야 하는 기술적 어려움이 있다.

따라서 본 연구는 U-Net의 기본 이론부터 다양한 변형 구조, 그리고 실제 의료 영상 모달리티(Modality)별 적용 사례를 종합적으로 검토하여, 연구자들이 자신의 문제에 적합한 모델을 선택하고 향후 연구 방향을 설정할 수 있도록 돕는 가이드라인을 제공하고자 한다.

## ✨ Key Contributions

본 논문의 핵심 기여는 U-Net의 진화 과정을 아키텍처 관점에서 분류하고, 이를 실제 의료 영상 적용 분야와 매핑하여 분석한 종합적인 서베이(Survey)를 제공한 점이다.

주요 분석 관점은 다음과 같다.
- **모듈성 및 가변성(Modularity and Mutability):** U-Net이 단순한 모델을 넘어, 다양한 딥러닝 기법(Attention, Residual, Dense connection 등)을 유연하게 통합할 수 있는 구조임을 강조한다.
- **변형 아키텍처의 체계적 분류:** Base U-Net부터 3D, Attention, Inception, Residual, Recurrent, Dense, U-Net++, Adversarial U-Net 및 직렬/병렬 구조까지 확장된 형태를 상세히 분석한다.
- **모달리티별 적용 맵 제공:** MRI, CT, 망막 저저(Fundus), 현미경(Microscopy), 피부경(Dermoscopy), 초음파(Ultrasound), X-ray 등 다양한 영상 데이터에서 어떤 변형 모델이 주로 사용되었는지 구체적인 문헌 근거를 통해 제시한다.

## 📎 Related Works

논문은 U-Net의 모태가 된 Long et al.의 Fully Convolutional Networks (FCN)를 언급하며, 기존의 분류 네트워크가 픽셀 수준의 문맥 정보를 제공하지 못하는 한계를 지적한다. 

또한, Ronneberger et al.이 제안한 오리지널 U-Net이 ISBI 2012 챌린지와 2015년 세포 추적 챌린지에서 압도적인 성능을 보이며 의료 영상 세그멘테이션의 새로운 기준이 되었음을 설명한다. 기존 접근 방식과의 차별점은 적은 양의 학습 데이터로도 고해상도 세그멘테이션 맵을 생성할 수 있다는 점이며, 이는 Random Elastic Deformation과 같은 데이터 증강 기법과 가중치 손실 함수(Weighted Loss Function)를 통해 달성되었다.

## 🛠️ Methodology

본 논문은 다양한 U-Net 변형 구조의 핵심 메커니즘을 다음과 같이 설명한다.

### 1. Base U-Net
전체 구조는 수축 경로(Contracting Path/Encoder)와 확장 경로(Expansive Path/Decoder)로 구성된 대칭적 U자 형태이다.
- **수축 경로:** 3x3 컨볼루션 $\rightarrow$ ReLU $\rightarrow$ Max-pooling의 반복을 통해 분류 정보를 추출한다.
- **확장 경로:** 2x2 Up-convolution을 통해 해상도를 높이며, 수축 경로에서 추출된 특징 맵을 Crop 하여 Concatenation(결합)함으로써 국소적 분류 정보를 학습한다.
- **에너지 함수:** 최종 특징 맵에 픽셀 단위 SoftMax 함수 $p_k$를 적용하여 세그멘테이션 결과를 생성한다.

### 2. 주요 변형 아키텍처 및 수식
- **3D U-Net:** 모든 2D 연산을 3D 컨볼루션, 3D Max-pooling, 3D Up-convolution으로 대체하여 볼륨 데이터(Volumetric data)를 처리한다.
- **Attention U-Net:** Attention Gate를 도입하여 불필요한 영역의 특징을 억제하고 중요한 객체에 집중한다. 가산적 주의 집중(Additive Attention) 메커니즘은 다음과 같이 정의된다.
  $$ \text{Attention Gate: } \sigma_1(W_x x_l + W_g g + b) $$
  $$ \text{Weighting: } \alpha = \sigma_2(\text{ReLU}(\dots)) $$
  여기서 $x_l$은 수축 경로의 특징이고 $g$는 게이팅 신호이며, $\sigma_2$는 Sigmoid 함수이다.
- **Inception U-Net:** 다양한 크기의 필터를 동일 레이어에 병렬로 배치하여 다양한 크기의 객체를 효과적으로 포착하며, 연산량 감소를 위해 1x1 컨볼루션을 통한 차원 축소를 수행한다.
- **Residual U-Net:** Skip connection을 통해 입력값을 출력값에 더함으로써 기울기 소실(Vanishing Gradient) 문제를 해결하고 더 깊은 네트워크를 가능하게 한다.
  $$ x_{l+1} = f(F(x_l) + h(x_l)) $$
  여기서 $F(\cdot)$는 잔차 함수, $h(\cdot)$는 항등 매핑(Identity mapping)이다.
- **Recurrent Convolutional Network (RCNN):** 피드백 루프를 통해 특징 맵을 반복적으로 업데이트하여 주변 문맥 정보를 더 잘 반영한다.
  $$ y = \text{activation}(w_f x_f(t) + w_r x_r(t-1) + b) $$
- **Dense U-Net:** DenseNet 블록을 사용하여 이전의 모든 레이어 특징 맵을 채널 방향으로 결합(Concatenation)함으로써 그래디언트 전파를 극대화한다.
  $$ x_l = H_l([x_0, x_1, \dots, x_{l-1}]) $$
- **U-Net++:** 수축 경로와 확장 경로 사이에 중첩된 Skip connection(Intermediary grid)을 배치하여 두 경로 간의 세만틱 갭(Semantic gap)을 줄인다.
- **Adversarial U-Net (GAN):** 생성자(Generator, U-Net 구조)와 판별자(Discriminator)가 경쟁하는 구조이다. 생성자는 실제 정답 맵과 구분이 안 될 정도의 정밀한 세그멘테이션 맵을 생성하는 것을 목표로 한다.
  $$ \min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{data}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))] $$

### 3. 기타 구조적 배치
- **Cascaded arrangement:** 여러 개의 U-Net을 직렬로 연결하여, 첫 번째 모델이 대략적인 영역(예: 간)을 찾고 두 번째 모델이 세부 객체(예: 종양)를 찾는 계층적 구조이다.
- **Parallel arrangement:** 여러 U-Net을 병렬로 배치하여 결과를 통합하거나, 2.5D U-Net과 같이 3개 축의 2D 투영 영상에서 각각 세그멘테이션을 수행한 뒤 융합하는 방식이다.

## 📊 Results

본 논문은 개별 실험 결과보다는 2017년부터 2020년 초까지 발표된 방대한 문헌을 분석한 통계적 결과를 제시한다.

- **가장 많이 사용된 모달리티:** MRI가 가장 높은 비중을 차지하며, 그 뒤를 CT, 망막 저저 영상이 잇고 있다.
- **주요 적용 분야:** 
    - **MRI:** 뇌종양 진단, 심혈관 구조 분석, 전립선암 진단 등에 광범위하게 적용되었다.
    - **CT:** 간암, 폐암 진단 및 복부 다기관 세그멘테이션에 주로 사용되었다.
    - **현미경 영상:** 세포 핵(Nuclei) 및 세포 경계 분리 작업에서 U-Net의 강력한 성능이 확인되었다.
- **정성적 결과:** U-Net과 그 변형 모델들이 단순한 세그멘테이션을 넘어 이미지 노이즈 제거(Denoising), 이미지 등록(Registration), 초해상도 복원(Super-resolution) 등 다른 정교한 작업에서도 가능성을 보였다.
- **COVID-19 대응:** 2020년 초, 흉부 CT 및 X-ray 영상을 이용한 COVID-19 조기 진단을 위해 U-Net 기반의 빠른 스크리닝 알고리즘들이 신속하게 개발 및 배포되었음을 확인하였다.

## 🧠 Insights & Discussion

### 강점 및 가능성
U-Net의 가장 큰 강점은 **극도의 모듈성(Modularity)**이다. 기본 구조를 유지하면서 내부 블록을 Residual, Dense, Inception 블록으로 교체하거나 Attention Gate를 추가하는 것만으로도 특정 의료 영상의 특성에 맞게 모델을 최적화할 수 있다. 이러한 유연성 덕분에 U-Net은 단순한 아키텍처를 넘어 하나의 '프레임워크'로서 기능하고 있다.

### 한계 및 해결 과제
1. **계산 자원 문제:** 모델이 깊어지고 복잡해짐에 따라 학습 시간이 증가하며, 이는 실시간 진단 시스템 적용에 걸림돌이 된다. Transfer Learning이나 EfficientNet과 같은 최적화 프레임워크 도입이 대안으로 제시된다.
2. **데이터 희소성:** 레이블링된 데이터의 부족은 여전한 문제이며, Random Deformation 외에도 GAN을 이용한 합성 데이터 생성이 해결책으로 논의되고 있다.
3. **블랙박스 문제 (Interpretability):** 딥러닝 모델 내부의 은닉층이 어떻게 작동하는지 알 수 없는 '블랙박스' 특성은 의료 현장에서의 신뢰성과 재현성 문제를 야기하며, 이는 실제 임상 시험 적용을 늦추는 주요 원인이 된다.

## 📌 TL;DR

본 논문은 의료 영상 세그멘테이션의 표준이 된 U-Net의 기본 이론과 이를 발전시킨 다양한 변형 모델(Attention, Residual, Dense, GAN 등) 및 배치 전략(Cascaded, Parallel)을 체계적으로 분석한 종합 보고서이다. U-Net은 뛰어난 모듈성 덕분에 MRI, CT 등 거의 모든 의료 영상 모달리티에 성공적으로 적용되었으며, 특히 데이터가 부족한 의료 환경에서 매우 효율적이다. 향후 연구는 모델의 해석 가능성(Explainability)을 높이고 계산 효율성을 최적화하는 방향으로 전개될 것으로 보인다.