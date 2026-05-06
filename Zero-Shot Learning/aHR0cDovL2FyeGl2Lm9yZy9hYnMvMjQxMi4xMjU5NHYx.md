# A Simple and Efficient Baseline for Zero-Shot Generative Classification

Zipeng Qi, Buhua Liu, Shiyan Zhang, Bao Li, Zhiqiang Xu, Haoyi Xiong, Zeke Xie (2024)

## 🧩 Problem to Solve

본 논문은 최근 주목받고 있는 대규모 Diffusion Model을 활용한 Zero-shot 분류기의 치명적인 단점인 **극도로 느린 추론 속도(Inference Speed)** 문제를 해결하고자 한다.

기존의 Diffusion 기반 Zero-shot 분류기들은 각 클래스에 대해 이미지의 Denoising Loss를 계산하는 방식을 사용한다. 이 방식은 ImageNet과 같은 대규모 데이터셋에서 단 한 장의 이미지를 분류하는 데 약 1,000초 이상의 시간이 소요되어 실제 서비스나 실시간 응용 분야에 적용하는 것이 사실상 불가능하다. 따라서 본 연구의 목표는 기존의 성능을 유지하거나 오히려 향상시키면서도, 실용적인 수준으로 추론 속도를 가속화한 효율적인 Zero-shot Generative Classifier를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 추론 시마다 반복되는 무거운 Diffusion 역과정(Reverse Process)을 제거하고, 이를 **사전 준비 단계(Preparation Phase)**로 옮기는 것이다.

구체적으로, 사전 학습된 Text-to-Image Diffusion Model로 클래스별 대표 이미지들을 생성하고, DINOv2 이미지 인코더를 통해 추출된 특징(Feature)들이 가우시안 분포(Gaussian Distribution)를 따른다는 점에 착안하여 **Gaussian Mixture Model (GMM)**을 통해 클래스를 모델링한다. 이를 통해 추론 단계에서는 단순한 가우시안 확률 계산만으로 분류를 수행함으로써 속도를 획기적으로 높였다.

## 📎 Related Works

- **Text-to-Image Diffusion Models**: GLIDE, Imagen, Stable Diffusion(SD) 등이 있으며, 주로 이미지 생성 작업에 집중되어 왔다. 본 논문은 이러한 생성 모델의 능력을 판별(Discriminative) 작업으로 확장했다.
- **Generative Classifiers**: 베이즈 관점에서 데이터 분포를 모델링하는 분류기로, 판별 모델보다 강건성(Robustness)이 높다는 연구가 많다. 하지만 대부분 학습 데이터가 필요하며, Zero-shot 설정에서의 연구는 부족한 상태였다.
- **Zero-Shot Classification**: CLIP과 같은 Vision-Language Model(VLM)이 대표적이다. 기존 Diffusion 기반 분류기(Li et al., Clark and Jaini)들은 Denoising Loss를 사용해 SOTA 성능을 냈으나, 앞서 언급한 속도 문제가 심각했다.

## 🛠️ Methodology

제안된 **Gaussian Diffusion Classifier (GDC)**는 크게 두 단계로 구성된다.

### 1. Preparation Phase (사전 준비 단계)

이 단계는 모델 배포 전 한 번만 수행되며, 모든 클래스에 대해 특징 분포를 미리 계산한다.

- **Reference Image Generation**: 각 클래스 $y_i$에 대해 Diffusion Model $M$을 사용하여 $N$장(예: 240장)의 참조 이미지를 생성한다. 이때 이미지의 다양성을 확보하기 위해 'a photo of a {}', 'an origami {}' 등 8가지의 Prompt Augmentation 템플릿을 사용한다.
- **Feature Extraction**: 생성된 이미지들을 DINOv2 인코더 $E$에 통과시켜 $d$-차원의 임베딩 벡터 $e$를 추출한다.
- **GMM Construction**: 각 클래스별 임베딩들의 평균 벡터 $\mu$와 공분산 행렬 $\Sigma$를 계산하여 가우시안 모델을 구축한다. 효율적인 계산을 위해 공분산 행렬의 역행렬인 정밀도 행렬(Precision Matrix) $\hat{\Sigma}^{-1} = (\Sigma + \epsilon I)^{-1}$을 구하고, 이에 대한 Cholesky Decomposition $LL^*$를 수행하여 저장한다.

### 2. Gaussian-based Classification Phase (분류 단계)

테스트 이미지 $x$가 입력되면 다음과 같은 절차로 분류를 수행한다.

- **Embedding Extraction**: 이미지 $x$를 DINOv2 인코더에 통해 임베딩 $e$를 한 번만 추출한다.
- **Probability Calculation**: 베이즈 정리를 이용하여 각 클래스 $y_i$에 속할 사후 확률 $p(y_i|x)$를 계산한다.

$$p(y_i|e) = \frac{p(e|y_i)p(y_i)}{\sum_{j=1}^{k} p(e|y_j)p(y_j)}$$

여기서 $p(e|y_i)$는 $d$-차원 공간에서의 가우시안 밀도 함수로 다음과 같이 정의된다.

$$p(e|y_i) = \frac{1}{(2\pi)^{d/2}(|\Sigma_i|)^{1/2}} \exp \left( -\frac{1}{2}(e-\mu_i)^T \hat{\Sigma}_i^{-1}(e-\mu_i) \right)$$

- **Label Prediction**: 가장 높은 확률을 가진 클래스를 최종 결과로 선택한다.
$$y_{pred} = \arg \max_{y_i} p(y_i|e)$$

## 📊 Results

### 실험 설정

- **Backbone**: SDXL-turbo (Diffusion Model), DINOv2 (Image Encoder).
- **데이터셋**: ImageNet-1K, CIFAR-10/100, Flower-102, Oxford Pet-37, Food-101, STL-10, DTD, Caltech-101.
- **지표**: Zero-shot Accuracy, 단일 이미지 분류 시간.

### 주요 결과

- **분류 정확도**: ImageNet 기준으로 기존 Loss-based Diffusion 분류기($61.40\%$)보다 약 10%p 향상된 **$71.44\%$**를 달성하였다. 평균적으로 여러 데이터셋에서 기존 방식보다 4.9~15.9%p 높은 성능을 보였다.
- **계산 효율성**: ImageNet 기준 단일 이미지 분류 시간이 $1,133$초에서 **$0.03$초**로 단축되었다. 이는 약 **30,000배 이상의 가속**을 의미한다.
- **One-shot 성능**: 참조 이미지 중 단 한 장을 실제 데이터로 교체했을 때, ImageNet에서 CLIP($75.2\%$)보다 높은 $76.2\%$의 정확도를 기록하며 매우 적은 데이터로도 성능 향상이 가능함을 보였다.
- **모델 확장성**: 더 강력한 Diffusion Model(SD 1.1 $\rightarrow$ SDXL-turbo)을 사용할수록 분류 정확도가 단조 증가하는 경향을 확인하였다.

## 🧠 Insights & Discussion

### 강점 및 분석

- **효율성과 정확도의 동시 확보**: 기존 연구들이 정확도를 위해 속도를 포기했다면, GDC는 가우시안 분포 모델링을 통해 두 마리 토끼를 모두 잡았다.
- **라벨 오류 수정 능력**: ImageNet의 일부 잘못된 라벨(Label Error)에 대해, GDC가 오히려 인간 주석자보다 더 정확한 시맨틱 라벨을 예측하는 사례가 발견되었다. 이는 생성 모델이 학습한 일반적인 개념이 데이터셋의 노이즈보다 강건할 수 있음을 시사한다.

### 한계 및 비판적 해석

- **실패 사례 분석**:
    1. 객체가 배경에 너무 잘 숨어 있는 경우(예: 물뱀) 환경 특징에 압도되어 오분류한다.
    2. 이미지 내에 여러 객체가 공존하는 경우, 주 객체가 무엇인지 판단하는 데 어려움이 있다.
    3. 매우 특이한 카메라 앵글의 경우, Diffusion Model이 생성하는 일반적인 뷰와 달라 인식률이 떨어진다.
- **가정의 단순함**: 모든 클래스의 특징 분포가 가우시안 분포를 따른다고 가정한 점은 단순하지만, 실제 데이터의 복잡한 분포를 완전히 캡처하지 못할 가능성이 있다. 저자 또한 이를 해결하기 위해 저차원 투영(Low-dimensional projection)을 제안하였다.

## 📌 TL;DR

본 논문은 Diffusion Model의 무거운 추론 과정을 사전 준비 단계의 GMM 구축으로 대체하여, **Zero-shot 분류 속도를 30,000배 가속화하면서도 정확도를 10%p 이상 향상**시킨 **GDC(Gaussian Diffusion Classifier)**를 제안한다. 이 연구는 생성 모델을 판별 작업에 효율적으로 활용하는 새로운 베이스라인을 제시하였으며, 향후 더 강력한 생성 모델이 등장함에 따라 분류 성능이 자연스럽게 향상될 수 있는 구조를 갖추고 있어 실용적 가치가 매우 높다.
