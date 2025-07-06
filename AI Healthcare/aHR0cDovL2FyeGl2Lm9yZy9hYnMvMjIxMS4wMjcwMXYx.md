# MONAI: An open-source framework for deep learning in healthcare
M. Jorge Cardoso, Wenqi Li, Richard Brown, Nic Ma, Eric Kerfoot, Yiheng Wang, Benjamin Murrey, Andriy Myronenko, Can Zhao, Dong Yang, Vishwesh Nath, Yufan He, Ziyue Xu, Ali Hatamizadeh, Andriy Myronenko, Wentao Zhu, Yun Liu, Mingxin Zheng, Yucheng Tang, Isaac Yang, Michael Zephyr, Behrooz Hashemian, Sachidanand Alle, Mohammad Zalbagi Darestani, Charlie Budd, Marc Modat, Tom Vercauteren, Guotai Wang, Yiwen Li, Yipeng Hu, Yunguan Fu, Benjamin Gorman, Hans Johnson, Brad Genereaux, Barbaros S. Erdal, Vikash Gupta, Andres Diaz-Pinto, Andre Dourson, Lena Maier-Hein, Paul F. Jaeger, Michael Baumgartner, Jayashree Kalpathy-Cramer, Mona Flores, Justin Kirby, Lee A.D. Cooper, Holger R. Roth, Daguang Xu, David Bericat, Ralf Floca, S. Kevin Zhou, Haris Shuaib, Keyvan Farahani, Klaus H. Maier-Hein, Stephen Aylward, Prerna Dogra, Sebastien Ourselin and Andrew Feng

## 🧩 Problem to Solve
의료 분야 인공지능(AI)은 질병 탐지, 진단, 예후 예측 및 임상 개입 개선에 지대한 잠재력을 가지고 있습니다. 그러나 AI 모델이 임상적으로 사용되기 위해서는 안전하고 재현 가능하며 견고해야 하며, 기본 소프트웨어 프레임워크는 처리되는 의료 데이터의 특수성(예: 기하학, 생리학, 물리학)을 인지해야 합니다. 기존의 범용 딥러닝 프레임워크(TensorFlow, PyTorch 등)는 이러한 의료 데이터 특화 기능을 충분히 지원하지 않아 개발 주기가 길어지고 위험이 증가합니다. 또한, 기존의 의료 특화 프레임워크들은 파편화되어 있어 개발 노력의 분산과 코드 품질 저하를 야기합니다. 따라서 의료 AI 모델 개발을 간소화하고 가속화하며, 임상 배포의 위험을 줄일 수 있는 통일되고 표준화된 프레임워크의 필요성이 대두되었습니다.

## ✨ Key Contributions
*   **MONAI Core 프레임워크 소개:** 의료 분야 딥러닝을 위한 PyTorch 기반의 오픈소스, 커뮤니티 지원 프레임워크인 MONAI(Medical Open Network for AI)를 제시합니다.
*   **의료 데이터 특화 기능 확장:** PyTorch를 확장하여 의료 영상 데이터에 특히 중점을 둔 목적별 AI 모델 아키텍처, 변환 및 유틸리티를 제공하여 의료 AI 모델 개발 및 배포를 간소화합니다.
*   **PyTorch 디자인 철학 계승:** PyTorch의 단순하고, 추가적이며, 구성 가능한 접근 방식을 유지하며, PyTorch 생태계와 긴밀하게 협력하여 높은 호환성과 사용 편의성을 제공합니다.
*   **광범위한 핵심 모듈 제공:** 데이터 처리(변환, 데이터셋), 모델 구성(네트워크 아키텍처), 학습 프로세스(손실 함수, 지표, 엔진, 핸들러), 시각화 및 유틸리티 등 의료 딥러닝 워크플로우의 모든 단계를 지원하는 포괄적인 모듈 세트를 포함합니다.
*   **재현성 및 품질 보장:** 소프트웨어 개발의 모범 사례를 따르며, 사용하기 쉽고, 견고하며, 잘 문서화되고, 테스트된 프레임워크를 제공하여 연구의 재현성과 코드 품질을 향상시킵니다.
*   **오픈소스 및 컨소시엄 기반 개발:** Apache-2.0 라이선스를 통해 연구 및 상업적 활용을 장려하고, 여러 대학 및 산업 파트너가 참여하는 컨소시엄을 통해 지속적인 개발과 커뮤니티 확장을 추진합니다.

## 📎 Related Works
*   **범용 딥러닝 프레임워크:** TensorFlow, Keras, PyTorch, JAX, Apache MXNet. (과거 프레임워크: Theano, Torch, Caffe, CNTK).
*   **의료 특화 딥러닝 프레임워크:** NiftyNet, DLTK, DeepNeuro, NVIDIA Clara, Microsoft Project InnerEye.
*   **의료 영상 처리 라이브러리:** ITK, Nibabel, PIL.
*   **PyTorch 기반 품질 개선 프레임워크:** PyTorch Ignite (고수준 엔진 및 추상화 제공).
*   **컴퓨터 비전 라이브러리:** Torchvision, Kornia.
*   **데이터 증강 라이브러리:** torchIO, BatchGenerator, Rising, cuCIM.
*   **의료 영상 등록 툴킷:** DeepReg.
*   **시각화 도구:** Tensorboard, Tensorboard-plugin-3d.
*   **의료 참조 데이터셋:** MedNIST, Medical Segmentation Decathlon, The Cancer Imaging Archive (TCIA).

## 🛠️ Methodology
MONAI는 PyTorch의 핵심 가이드라인을 따르면서 의료 딥러닝의 고유한 요구 사항을 충족하도록 설계되었습니다.

1.  **PyTorch 디자인 원칙 준수:**
    *   **PyTorch와 유사한 경험:** PyTorch 사용자가 최소한의 학습 곡선으로 MONAI를 활용할 수 있도록 PyTorch의 코딩 스타일과 API 설계를 따릅니다.
    *   **점진적이고 선택적인 확장:** PyTorch 기능을 완전히 재구축하는 대신, 기존 PyTorch 워크플로우에 개별 MONAI 구성 요소(변환, 레이어, 손실 함수 등)를 점진적으로 통합할 수 있도록 설계되었습니다.
    *   **PyTorch 생태계 협력:** PyTorch Ignite와 같은 기존 PyTorch 기반 프레임워크를 확장하고, 다른 의료 특화 라이브러리(torchIO, Kornia 등)와의 호환성을 위한 어댑터 도구를 제공하여 시너지를 창출합니다.

2.  **시스템 구성 요소:**
    *   `monai.data`: 데이터셋, 데이터 로더, 리더/라이터, 합성 데이터 생성 도구.
    *   `monai.losses`: 의료 영상 특화 손실 함수(Dice 및 파생 모델, Focal Loss, Tversky Loss 등).
    *   `monai.networks`: 다양한 참조 네트워크 아키텍처(UNet, ResNet, EfficientNet, ViT 등)와 일반 목적 네트워크 구현. 1D/2D/3D 입력 및 출력에 유연하게 대응합니다.
    *   `monai.transforms`: 의료 영상의 I/O, 전처리 및 증강을 위한 포괄적인 변환 세트. 물리 기반 변환(k-space 노이즈), 역변환(Invertible transforms), 배열/딕셔너리 기반 변환, CPU/GPU 호환성을 지원합니다.
    *   `monai.engines` 및 `monai.handlers`: PyTorch Ignite 기반의 학습 및 평가 워크플로우 엔진과 이벤트 핸들러를 제공하여 학습 루프를 추상화하고 재현성을 높입니다.
    *   `monai.metrics`: 모델 성능 평가를 위한 다양한 지표(Metrics Reloaded 컨소시엄 권장 지표 포함)와 분석 도구.
    *   `monai.visualize`: Tensorboard 통합, 3D 영상 렌더링, 이미지/라벨 블렌딩 등 시각화 유틸리티.
    *   `MetaTensor`: `torch.Tensor`를 상속하여 이미지 데이터와 함께 메타데이터 및 적용된 변환 이력을 저장함으로써, 데이터의 기하학적 정보 보존 및 역변환을 용이하게 합니다.
    *   `CacheDataset` 및 `PersistentDataset`: 대규모 의료 데이터셋의 전처리 효율성을 높이기 위해, 결정론적 변환 결과를 메모리 또는 파일 시스템에 캐싱하는 기능을 제공합니다.

3.  **학습 및 추론 워크플로우:**
    *   `SupervisedTrainer` 및 `SupervisedEvaluator`를 사용하여 표준 PyTorch 학습/평가 루프를 간소화합니다.
    *   분산 데이터 병렬(Distributed data-parallel) 학습을 지원하여 다중 GPU/노드 환경에서 효율적인 학습이 가능합니다.
    *   대용량 3D 영상 추론을 위한 슬라이딩 윈도우(Sliding Window) 방식을 제공하며, 메모리 제약 속에서도 고성능을 달성합니다.
    *   `set_determinism` 유틸리티를 통해 무작위 프로세스의 재현성을 보장하여 실험 결과의 일관성을 유지합니다.
    *   Occlusion sensitivity, GradCAM, Smoothgrad 등 모델 해석(Interpretability) 도구를 내장하여 모델의 의사결정 과정을 시각적으로 이해할 수 있도록 돕습니다.

## 📊 Results
MONAI는 다양한 의료 딥러닝 애플리케이션에서 플랫폼의 효율성과 유용성을 입증했습니다.

*   **학습 워크플로우 간소화 및 재현성 향상:** PyTorch의 저수준 학습 루프를 MONAI의 `SupervisedTrainer`와 `SupervisedEvaluator` 같은 고수준 추상화로 대체함으로써, 개발자가 직접 구현해야 하는 코드의 양을 획기적으로 줄이고 오류 발생 가능성을 낮췄습니다. 이는 연구 재현성을 크게 향상시킵니다.
*   **결정론적 결과 보장:** `monai.utils.set_determinism` 기능을 통해 무작위 시드 설정과 결정론적 알고리즘 사용을 지원함으로써, 무작위 변환이나 네트워크의 드롭아웃(dropout)과 같은 확률적 요소가 포함된 실험에서도 일관된 결과를 얻을 수 있음을 보여주었습니다. 이는 분류 모델의 행동 해석 등에서 매우 유용합니다.
*   **의료 영상 등록(Registration) 지원:** DeepReg 툴킷에서 포팅된 손실 함수(예: Dice loss, bending energy)와 변형 필드 추정을 위한 MONAI 클래스를 활용하여 폐 CT 영상 등록과 같은 복잡한 의료 영상 등록 문제를 효과적으로 해결할 수 있음을 입증했습니다. PyTorch에는 없는 삼차 보간(tricubic interpolation)과 같은 고급 보간 모델을 지원하여 정확도와 속도를 동시에 확보합니다.
*   **성능 최적화 및 GPU 활용 극대화:** `CacheDataset`와 `ToDeviced` 변환을 함께 사용하여 데이터를 GPU 메모리에 직접 캐싱하는 효율적인 데이터 로딩 및 전처리 방식을 제안하고 구현했습니다. 이를 통해 매 에포크마다 CPU-GPU 간 데이터 복사 오버헤드를 줄여 학습 속도를 크게 향상시켰습니다. 또한, 자동 혼합 정밀도(AMP), 분산 데이터 병렬(Distributed Data Parallel)과 같은 PyTorch의 고급 GPU 가속 기능을 지원하며, 파이프라인 프로파일링 및 GPU 활용 최적화에 대한 상세 가이드를 제공하여 사용자들의 성능 향상을 돕습니다.
*   **광범위한 의료 AI 응용:** 분할(Segmentation), 분류(Classification), 등록(Registration) 등 의료 영상 분야의 핵심 문제를 MONAI 프레임워크를 활용하여 성공적으로 구현하고 해결할 수 있음을 보여주었습니다. 이는 MONAI가 다양한 임상 및 연구 시나리오에 적용 가능한 유연하고 강력한 도구임을 입증합니다.

## 🧠 Insights & Discussion
MONAI는 의료 딥러닝 분야의 고유한 도전을 해결하기 위한 핵심적인 통찰력을 제공하며, 다음과 같은 광범위한 함의를 가집니다.

*   **의료 AI 개발의 가속화 및 간소화:** MONAI는 의료 영상 데이터의 복잡한 특성(고차원성, 풍부한 메타데이터, 특정 변환 요구사항)을 PyTorch 위에 계층적으로 추상화하여, 연구자들이 낮은 수준의 구현 상세에 얽매이지 않고 혁신적인 알고리즘 개발에 집중할 수 있도록 돕습니다. 이는 의료 AI 모델의 연구 개발 주기를 크게 단축하고 접근성을 높입니다.
*   **연구 재현성 및 임상 신뢰도 향상:** 표준화된 컴포넌트, 잘 정의된 워크플로우, 그리고 결정론적 동작 지원은 코드의 오류 가능성을 줄이고 실험 결과의 일관성과 재현성을 보장합니다. 이는 특히 안전성과 견고성이 중요한 임상 환경에 AI 모델을 도입할 때 필수적인 신뢰도를 구축하는 데 기여합니다.
*   **강력한 커뮤니티 및 생태계 구축:** Apache-2.0 라이선스 기반의 오픈소스 전략과 다양한 기관이 참여하는 컨소시엄 구조는 전 세계 연구자, 임상의, 산업 파트너의 적극적인 기여를 유도합니다. 이러한 협력은 MONAI의 지속적인 품질 개선, 기능 확장, 그리고 의료 AI 분야의 사실상 표준 프레임워크로의 성장을 촉진합니다. MONAI Label, MONAI Deploy, MONAI FL 등 관련 프로젝트들의 확장 또한 의료 AI의 전반적인 라이프사이클을 지원하는 포괄적인 생태계를 구축하고 있습니다.
*   **임상 배포 용이성 증대:** Torchscript 호환성을 통해 학습된 모델을 Python 의존성 없이 배포할 수 있도록 하는 기능은 AI 모델을 실제 임상 시스템에 통합하는 과정의 복잡성을 크게 줄여줍니다. 이는 연구 결과가 실질적인 임상적 가치로 이어지는 데 중요한 역할을 합니다.
*   **제약 사항 및 미래 방향:** 현재 MONAI는 주로 영상 데이터에 초점을 맞추고 있지만, HL7 FIHR, EEG 신호와 같은 다른 형태의 정형 데이터 지원으로 확장할 계획입니다. 또한, Metrics Reloaded와 같은 컨소시엄과의 지속적인 협력을 통해 측정 지표의 신뢰성을 더욱 향상시키고, The Cancer Imaging Archive(TCIA)와 같은 대규모 연구 데이터셋을 통합하여 커뮤니티의 데이터 접근성을 높이는 방안을 모색하고 있습니다.

## 📌 TL;DR
MONAI는 PyTorch 기반의 오픈소스 의료 딥러닝 프레임워크로, 일반 딥러닝 프레임워크가 놓치는 의료 데이터의 특수성(예: 고차원 3D 영상, 복잡한 메타데이터)을 해결하는 것을 목표로 합니다. PyTorch의 철학을 계승하여 의료 특화 데이터 변환, 손실 함수, 네트워크 아키텍처, 효율적인 데이터 처리 및 학습/추론 엔진을 제공합니다. 이를 통해 의료 AI 모델 개발을 간소화하고 가속화하며, 연구 재현성을 높이고, 궁극적으로 AI 모델의 임상 배포를 용이하게 합니다.