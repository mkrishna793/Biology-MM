// =============================================================================
// Trium-XBT Portable Vulkan Compute Backend (Zero-SDK Dynamic Loader)
// =============================================================================
// Loads libvulkan / libMoltenVK / vulkan-1.dll dynamically at runtime with
// zero compile-time dependency on Vulkan SDK headers.
//
// Features:
//   - Probes vulkan-1.dll / libvulkan.so.1 / libvulkan.so / libvulkan.1.dylib / libMoltenVK.dylib
//   - Enumerates physical devices (discrete > integrated > virtual > CPU)
//   - Queries VkPhysicalDeviceSubgroupProperties for adaptive WORKGROUP_SIZE
//     specialization constant tuning (32 vs 64)
//   - Queries VkPhysicalDeviceMemoryProperties for optimal device-local vs
//     host-visible staging transfers (with zero-copy fast path on UMA GPUs)
//   - Dispatches SPIR-V compute kernels for:
//       1) xbt_lj_dsf_csr.spv    (Full Symmetric CSR LJ + DSF Coulomb)
//       2) xbt_ternary_matvec.spv (Dual-Rail Bit-Sliced POPCNT MatVec)
//       3) xbt_msm_stencil27.spv  (Periodic 27-Point Radix-3 MSM Stencil)
//   - Returns xbt_vulkan_available() == 0 gracefully when no Vulkan ICD exists.
// =============================================================================

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#define XBT_EXPORT extern "C" __declspec(dllexport)
#else
#include <dlfcn.h>
#define XBT_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace {

inline void* vk_dlopen(const char* path) {
#if defined(_WIN32)
    return reinterpret_cast<void*>(LoadLibraryA(path));
#else
    return dlopen(path, RTLD_NOW | RTLD_LOCAL);
#endif
}

inline void* vk_dlsym(void* handle, const char* symbol) {
#if defined(_WIN32)
    return reinterpret_cast<void*>(GetProcAddress(reinterpret_cast<HMODULE>(handle), symbol));
#else
    return dlsym(handle, symbol);
#endif
}

inline void vk_dlclose(void* handle) {
#if defined(_WIN32)
    if (handle) FreeLibrary(reinterpret_cast<HMODULE>(handle));
#else
    if (handle) dlclose(handle);
#endif
}

} // namespace

// =============================================================================
// Minimal Standard Vulkan 1.1+ C ABI Definitions (Zero Vulkan SDK Dependency)
// =============================================================================
namespace xbt_vk_abi {

using VkBool32 = uint32_t;
using VkFlags = uint32_t;
using VkDeviceSize = uint64_t;
using VkResult = int32_t;
using VkStructureType = int32_t;
using VkSampleCountFlags = VkFlags;
using VkQueueFlags = VkFlags;
using VkShaderStageFlags = VkFlags;
using VkSubgroupFeatureFlags = VkFlags;
using VkBufferUsageFlags = VkFlags;
using VkMemoryPropertyFlags = VkFlags;
using VkAccessFlags = VkFlags;
using VkPipelineStageFlags = VkFlags;

constexpr VkResult VK_SUCCESS = 0;
constexpr VkResult VK_INCOMPLETE = 5;
constexpr uint32_t VK_API_VERSION_1_1 = (1u << 22) | (1u << 12);
constexpr uint64_t VK_WHOLE_SIZE = ~uint64_t(0);

constexpr VkStructureType VK_STRUCTURE_TYPE_APPLICATION_INFO = 0;
constexpr VkStructureType VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO = 1;
constexpr VkStructureType VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO = 2;
constexpr VkStructureType VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO = 3;
constexpr VkStructureType VK_STRUCTURE_TYPE_SUBMIT_INFO = 4;
constexpr VkStructureType VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO = 5;
constexpr VkStructureType VK_STRUCTURE_TYPE_FENCE_CREATE_INFO = 8;
constexpr VkStructureType VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO = 12;
constexpr VkStructureType VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO = 16;
constexpr VkStructureType VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO = 18;
constexpr VkStructureType VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO = 29;
constexpr VkStructureType VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO = 30;
constexpr VkStructureType VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO = 32;
constexpr VkStructureType VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO = 33;
constexpr VkStructureType VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO = 34;
constexpr VkStructureType VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET = 35;
constexpr VkStructureType VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO = 39;
constexpr VkStructureType VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO = 40;
constexpr VkStructureType VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO = 42;
constexpr VkStructureType VK_STRUCTURE_TYPE_MEMORY_BARRIER = 46;
constexpr VkStructureType VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 = 1000059001;
constexpr VkStructureType VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES = 1000094000;

constexpr uint32_t VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR = 0x00000001u;
constexpr VkQueueFlags VK_QUEUE_COMPUTE_BIT = 0x00000002u;
constexpr VkShaderStageFlags VK_SHADER_STAGE_COMPUTE_BIT = 0x00000020u;
constexpr uint32_t VK_PIPELINE_BIND_POINT_COMPUTE = 1u;
constexpr uint32_t VK_SHARING_MODE_EXCLUSIVE = 0u;
constexpr uint32_t VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER = 6u;
constexpr uint32_t VK_DESCRIPTOR_TYPE_STORAGE_BUFFER = 7u;
constexpr uint32_t VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT = 0x00000002u;
constexpr uint32_t VK_COMMAND_BUFFER_LEVEL_PRIMARY = 0u;
constexpr uint32_t VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT = 0x00000001u;
constexpr uint32_t VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT = 0x00000001u;

constexpr VkBufferUsageFlags VK_BUFFER_USAGE_TRANSFER_SRC_BIT = 0x00000001u;
constexpr VkBufferUsageFlags VK_BUFFER_USAGE_TRANSFER_DST_BIT = 0x00000002u;
constexpr VkBufferUsageFlags VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT = 0x00000010u;
constexpr VkBufferUsageFlags VK_BUFFER_USAGE_STORAGE_BUFFER_BIT = 0x00000020u;

constexpr VkMemoryPropertyFlags VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT = 0x00000001u;
constexpr VkMemoryPropertyFlags VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT = 0x00000002u;
constexpr VkMemoryPropertyFlags VK_MEMORY_PROPERTY_HOST_COHERENT_BIT = 0x00000004u;

constexpr VkAccessFlags VK_ACCESS_SHADER_READ_BIT = 0x00000020u;
constexpr VkAccessFlags VK_ACCESS_SHADER_WRITE_BIT = 0x00000040u;
constexpr VkAccessFlags VK_ACCESS_TRANSFER_READ_BIT = 0x00000800u;

constexpr VkPipelineStageFlags VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT = 0x00000800u;
constexpr VkPipelineStageFlags VK_PIPELINE_STAGE_TRANSFER_BIT = 0x00001000u;

constexpr uint32_t VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU = 1u;
constexpr uint32_t VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU = 2u;
constexpr uint32_t VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU = 3u;
constexpr uint32_t VK_PHYSICAL_DEVICE_TYPE_CPU = 4u;

// Dispatchable handles
struct VkInstance_T;
struct VkPhysicalDevice_T;
struct VkDevice_T;
struct VkQueue_T;
struct VkCommandBuffer_T;
using VkInstance = VkInstance_T*;
using VkPhysicalDevice = VkPhysicalDevice_T*;
using VkDevice = VkDevice_T*;
using VkQueue = VkQueue_T*;
using VkCommandBuffer = VkCommandBuffer_T*;

// Non-dispatchable handles (64-bit integer ABI across LP64 / LLP64)
using VkBuffer = uint64_t;
using VkDeviceMemory = uint64_t;
using VkShaderModule = uint64_t;
using VkDescriptorSetLayout = uint64_t;
using VkPipelineLayout = uint64_t;
using VkPipeline = uint64_t;
using VkPipelineCache = uint64_t;
using VkDescriptorPool = uint64_t;
using VkDescriptorSet = uint64_t;
using VkCommandPool = uint64_t;
using VkFence = uint64_t;
using VkSemaphore = uint64_t;
using VkSampler = uint64_t;
using VkBufferView = uint64_t;

struct VkExtent3D {
    uint32_t width;
    uint32_t height;
    uint32_t depth;
};

struct VkApplicationInfo {
    VkStructureType sType;
    const void* pNext;
    const char* pApplicationName;
    uint32_t applicationVersion;
    const char* pEngineName;
    uint32_t engineVersion;
    uint32_t apiVersion;
};

struct VkInstanceCreateInfo {
    VkStructureType sType;
    const void* pNext;
    VkFlags flags;
    const VkApplicationInfo* pApplicationInfo;
    uint32_t enabledLayerCount;
    const char* const* ppEnabledLayerNames;
    uint32_t enabledExtensionCount;
    const char* const* ppEnabledExtensionNames;
};

struct VkPhysicalDeviceLimits {
    uint32_t maxImageDimension1D;
    uint32_t maxImageDimension2D;
    uint32_t maxImageDimension3D;
    uint32_t maxImageDimensionCube;
    uint32_t maxImageArrayLayers;
    uint32_t maxTexelBufferElements;
    uint32_t maxUniformBufferRange;
    uint32_t maxStorageBufferRange;
    uint32_t maxPushConstantsSize;
    uint32_t maxMemoryAllocationCount;
    uint32_t maxSamplerAllocationCount;
    VkDeviceSize bufferImageGranularity;
    VkDeviceSize sparseAddressSpaceSize;
    uint32_t maxBoundDescriptorSets;
    uint32_t maxPerStageDescriptorSamplers;
    uint32_t maxPerStageDescriptorUniformBuffers;
    uint32_t maxPerStageDescriptorStorageBuffers;
    uint32_t maxPerStageDescriptorSampledImages;
    uint32_t maxPerStageDescriptorStorageImages;
    uint32_t maxPerStageDescriptorInputAttachments;
    uint32_t maxPerStageResources;
    uint32_t maxDescriptorSetSamplers;
    uint32_t maxDescriptorSetUniformBuffers;
    uint32_t maxDescriptorSetUniformBuffersDynamic;
    uint32_t maxDescriptorSetStorageBuffers;
    uint32_t maxDescriptorSetStorageBuffersDynamic;
    uint32_t maxDescriptorSetSampledImages;
    uint32_t maxDescriptorSetStorageImages;
    uint32_t maxDescriptorSetInputAttachments;
    uint32_t maxVertexInputAttributes;
    uint32_t maxVertexInputBindings;
    uint32_t maxVertexInputAttributeOffset;
    uint32_t maxVertexInputBindingStride;
    uint32_t maxVertexOutputComponents;
    uint32_t maxTessellationGenerationLevel;
    uint32_t maxTessellationPatchSize;
    uint32_t maxTessellationControlPerVertexInputComponents;
    uint32_t maxTessellationControlPerVertexOutputComponents;
    uint32_t maxTessellationControlPerPatchOutputComponents;
    uint32_t maxTessellationControlTotalOutputComponents;
    uint32_t maxTessellationEvaluationInputComponents;
    uint32_t maxTessellationEvaluationOutputComponents;
    uint32_t maxGeometryShaderInvocations;
    uint32_t maxGeometryInputComponents;
    uint32_t maxGeometryOutputComponents;
    uint32_t maxGeometryOutputVertices;
    uint32_t maxGeometryTotalOutputComponents;
    uint32_t maxFragmentInputComponents;
    uint32_t maxFragmentOutputAttachments;
    uint32_t maxFragmentDualSrcAttachments;
    uint32_t maxFragmentCombinedOutputResources;
    uint32_t maxComputeSharedMemorySize;
    uint32_t maxComputeWorkGroupCount[3];
    uint32_t maxComputeWorkGroupInvocations;
    uint32_t maxComputeWorkGroupSize[3];
    uint32_t subPixelPrecisionBits;
    uint32_t subTexelPrecisionBits;
    uint32_t mipmapPrecisionBits;
    uint32_t maxDrawIndexedIndexValue;
    uint32_t maxDrawIndirectCount;
    float maxSamplerLodBias;
    float maxSamplerAnisotropy;
    uint32_t maxViewports;
    uint32_t maxViewportDimensions[2];
    float viewportBoundsRange[2];
    uint32_t viewportSubPixelBits;
    size_t minMemoryMapAlignment;
    VkDeviceSize minTexelBufferOffsetAlignment;
    VkDeviceSize minUniformBufferOffsetAlignment;
    VkDeviceSize minStorageBufferOffsetAlignment;
    int32_t minTexelOffset;
    uint32_t maxTexelOffset;
    int32_t minTexelGatherOffset;
    uint32_t maxTexelGatherOffset;
    float minInterpolationOffset;
    float maxInterpolationOffset;
    uint32_t subPixelInterpolationOffsetBits;
    uint32_t maxFramebufferWidth;
    uint32_t maxFramebufferHeight;
    uint32_t maxFramebufferLayers;
    VkSampleCountFlags framebufferColorSampleCounts;
    VkSampleCountFlags framebufferDepthSampleCounts;
    VkSampleCountFlags framebufferStencilSampleCounts;
    VkSampleCountFlags framebufferNoAttachmentsSampleCounts;
    uint32_t maxColorAttachments;
    VkSampleCountFlags sampledImageColorSampleCounts;
    VkSampleCountFlags sampledImageIntegerSampleCounts;
    VkSampleCountFlags sampledImageDepthSampleCounts;
    VkSampleCountFlags sampledImageStencilSampleCounts;
    VkSampleCountFlags storageImageSampleCounts;
    uint32_t maxSampleMaskWords;
    VkBool32 timestampComputeAndGraphics;
    float timestampPeriod;
    uint32_t maxClipDistances;
    uint32_t maxCullDistances;
    uint32_t maxCombinedClipAndCullDistances;
    uint32_t discreteQueuePriorities;
    float pointSizeRange[2];
    float lineWidthRange[2];
    float pointSizeGranularity;
    float lineWidthGranularity;
    VkBool32 strictLines;
    VkBool32 standardSampleLocations;
    VkDeviceSize optimalBufferCopyOffsetAlignment;
    VkDeviceSize optimalBufferCopyRowPitchAlignment;
    VkDeviceSize nonCoherentAtomSize;
};

struct VkPhysicalDeviceSparseProperties {
    VkBool32 residencyStandard2DBlockShape;
    VkBool32 residencyStandard2DMultisampleBlockShape;
    VkBool32 residencyStandard3DBlockShape;
    VkBool32 residencyAlignedMipSize;
    VkBool32 residencyNonResidentStrict;
};

struct VkPhysicalDeviceProperties {
    uint32_t apiVersion;
    uint32_t driverVersion;
    uint32_t vendorID;
    uint32_t deviceID;
    uint32_t deviceType;
    char deviceName[256];
    uint8_t pipelineCacheUUID[16];
    VkPhysicalDeviceLimits limits;
    VkPhysicalDeviceSparseProperties sparseProperties;
};

struct VkPhysicalDeviceProperties2 {
    VkStructureType sType;
    void* pNext;
    VkPhysicalDeviceProperties properties;
};

struct VkPhysicalDeviceSubgroupProperties {
    VkStructureType sType;
    void* pNext;
    uint32_t subgroupSize;
    VkShaderStageFlags supportedStages;
    VkSubgroupFeatureFlags supportedOperations;
    VkBool32 quadOperationsInAllStages;
};

struct VkMemoryType {
    VkMemoryPropertyFlags propertyFlags;
    uint32_t heapIndex;
};

struct VkMemoryHeap {
    VkDeviceSize size;
    VkFlags flags;
};

struct VkPhysicalDeviceMemoryProperties {
    uint32_t memoryTypeCount;
    VkMemoryType memoryTypes[32];
    uint32_t memoryHeapCount;
    VkMemoryHeap memoryHeaps[16];
};

struct VkQueueFamilyProperties {
    VkQueueFlags queueFlags;
    uint32_t queueCount;
    uint32_t timestampValidBits;
    VkExtent3D minImageTransferGranularity;
};

struct VkDeviceQueueCreateInfo {
    VkStructureType sType;
    const void* pNext;
    VkFlags flags;
    uint32_t queueFamilyIndex;
    uint32_t queueCount;
    const float* pQueuePriorities;
};

struct VkDeviceCreateInfo {
    VkStructureType sType;
    const void* pNext;
    VkFlags flags;
    uint32_t queueCreateInfoCount;
    const VkDeviceQueueCreateInfo* pQueueCreateInfos;
    uint32_t enabledLayerCount;
    const char* const* ppEnabledLayerNames;
    uint32_t enabledExtensionCount;
    const char* const* ppEnabledExtensionNames;
    const void* pEnabledFeatures;
};

struct VkBufferCreateInfo {
    VkStructureType sType;
    const void* pNext;
    VkFlags flags;
    VkDeviceSize size;
    VkBufferUsageFlags usage;
    uint32_t sharingMode;
    uint32_t queueFamilyIndexCount;
    const uint32_t* pQueueFamilyIndices;
};

struct VkMemoryRequirements {
    VkDeviceSize size;
    VkDeviceSize alignment;
    uint32_t memoryTypeBits;
};

struct VkMemoryAllocateInfo {
    VkStructureType sType;
    const void* pNext;
    VkDeviceSize allocationSize;
    uint32_t memoryTypeIndex;
};

struct VkShaderModuleCreateInfo {
    VkStructureType sType;
    const void* pNext;
    VkFlags flags;
    size_t codeSize;
    const uint32_t* pCode;
};

struct VkDescriptorSetLayoutBinding {
    uint32_t binding;
    uint32_t descriptorType;
    uint32_t descriptorCount;
    VkShaderStageFlags stageFlags;
    const VkSampler* pImmutableSamplers;
};

struct VkDescriptorSetLayoutCreateInfo {
    VkStructureType sType;
    const void* pNext;
    VkFlags flags;
    uint32_t bindingCount;
    const VkDescriptorSetLayoutBinding* pBindings;
};

struct VkPushConstantRange {
    VkShaderStageFlags stageFlags;
    uint32_t offset;
    uint32_t size;
};

struct VkPipelineLayoutCreateInfo {
    VkStructureType sType;
    const void* pNext;
    VkFlags flags;
    uint32_t setLayoutCount;
    const VkDescriptorSetLayout* pSetLayouts;
    uint32_t pushConstantRangeCount;
    const VkPushConstantRange* pPushConstantRanges;
};

struct VkSpecializationMapEntry {
    uint32_t constantID;
    uint32_t offset;
    size_t size;
};

struct VkSpecializationInfo {
    uint32_t mapEntryCount;
    const VkSpecializationMapEntry* pMapEntries;
    size_t dataSize;
    const void* pData;
};

struct VkPipelineShaderStageCreateInfo {
    VkStructureType sType;
    const void* pNext;
    VkFlags flags;
    VkShaderStageFlags stage;
    VkShaderModule module;
    const char* pName;
    const VkSpecializationInfo* pSpecializationInfo;
};

struct VkComputePipelineCreateInfo {
    VkStructureType sType;
    const void* pNext;
    VkFlags flags;
    VkPipelineShaderStageCreateInfo stage;
    VkPipelineLayout layout;
    VkPipeline basePipelineHandle;
    int32_t basePipelineIndex;
};

struct VkDescriptorPoolSize {
    uint32_t type;
    uint32_t descriptorCount;
};

struct VkDescriptorPoolCreateInfo {
    VkStructureType sType;
    const void* pNext;
    VkFlags flags;
    uint32_t maxSets;
    uint32_t poolSizeCount;
    const VkDescriptorPoolSize* pPoolSizes;
};

struct VkDescriptorSetAllocateInfo {
    VkStructureType sType;
    const void* pNext;
    VkDescriptorPool descriptorPool;
    uint32_t descriptorSetCount;
    const VkDescriptorSetLayout* pSetLayouts;
};

struct VkDescriptorBufferInfo {
    VkBuffer buffer;
    VkDeviceSize offset;
    VkDeviceSize range;
};

struct VkWriteDescriptorSet {
    VkStructureType sType;
    const void* pNext;
    VkDescriptorSet dstSet;
    uint32_t dstBinding;
    uint32_t dstArrayElement;
    uint32_t descriptorCount;
    uint32_t descriptorType;
    const void* pImageInfo;
    const VkDescriptorBufferInfo* pBufferInfo;
    const VkBufferView* pTexelBufferView;
};

struct VkCommandPoolCreateInfo {
    VkStructureType sType;
    const void* pNext;
    VkFlags flags;
    uint32_t queueFamilyIndex;
};

struct VkCommandBufferAllocateInfo {
    VkStructureType sType;
    const void* pNext;
    VkCommandPool commandPool;
    uint32_t level;
    uint32_t commandBufferCount;
};

struct VkCommandBufferBeginInfo {
    VkStructureType sType;
    const void* pNext;
    VkFlags flags;
    const void* pInheritanceInfo;
};

struct VkBufferCopy {
    VkDeviceSize srcOffset;
    VkDeviceSize dstOffset;
    VkDeviceSize size;
};

struct VkMemoryBarrier {
    VkStructureType sType;
    const void* pNext;
    VkAccessFlags srcAccessMask;
    VkAccessFlags dstAccessMask;
};

struct VkSubmitInfo {
    VkStructureType sType;
    const void* pNext;
    uint32_t waitSemaphoreCount;
    const VkSemaphore* pWaitSemaphores;
    const VkPipelineStageFlags* pWaitDstStageMask;
    uint32_t commandBufferCount;
    const VkCommandBuffer* pCommandBuffers;
    uint32_t signalSemaphoreCount;
    const VkSemaphore* pSignalSemaphores;
};

struct VkFenceCreateInfo {
    VkStructureType sType;
    const void* pNext;
    VkFlags flags;
};

// Verify exact standard Vulkan ABI sizes on LP64
static_assert(sizeof(VkPhysicalDeviceLimits) == 504, "VkPhysicalDeviceLimits ABI mismatch");
static_assert(sizeof(VkPhysicalDeviceProperties) == 824, "VkPhysicalDeviceProperties ABI mismatch");
static_assert(sizeof(VkPhysicalDeviceMemoryProperties) == 520, "VkPhysicalDeviceMemoryProperties ABI mismatch");

// Function pointer types
using PFN_vkVoidFunction = void (*)();
using PFN_vkGetInstanceProcAddr = PFN_vkVoidFunction (*)(VkInstance, const char*);
using PFN_vkCreateInstance = VkResult (*)(const VkInstanceCreateInfo*, const void*, VkInstance*);
using PFN_vkDestroyInstance = void (*)(VkInstance, const void*);
using PFN_vkEnumeratePhysicalDevices = VkResult (*)(VkInstance, uint32_t*, VkPhysicalDevice*);
using PFN_vkGetPhysicalDeviceProperties = void (*)(VkPhysicalDevice, VkPhysicalDeviceProperties*);
using PFN_vkGetPhysicalDeviceProperties2 = void (*)(VkPhysicalDevice, VkPhysicalDeviceProperties2*);
using PFN_vkGetPhysicalDeviceMemoryProperties = void (*)(VkPhysicalDevice, VkPhysicalDeviceMemoryProperties*);
using PFN_vkGetPhysicalDeviceQueueFamilyProperties = void (*)(VkPhysicalDevice, uint32_t*, VkQueueFamilyProperties*);
using PFN_vkCreateDevice = VkResult (*)(VkPhysicalDevice, const VkDeviceCreateInfo*, const void*, VkDevice*);
using PFN_vkDestroyDevice = void (*)(VkDevice, const void*);
using PFN_vkGetDeviceQueue = void (*)(VkDevice, uint32_t, uint32_t, VkQueue*);
using PFN_vkCreateBuffer = VkResult (*)(VkDevice, const VkBufferCreateInfo*, const void*, VkBuffer*);
using PFN_vkDestroyBuffer = void (*)(VkDevice, VkBuffer, const void*);
using PFN_vkGetBufferMemoryRequirements = void (*)(VkDevice, VkBuffer, VkMemoryRequirements*);
using PFN_vkAllocateMemory = VkResult (*)(VkDevice, const VkMemoryAllocateInfo*, const void*, VkDeviceMemory*);
using PFN_vkFreeMemory = void (*)(VkDevice, VkDeviceMemory, const void*);
using PFN_vkBindBufferMemory = VkResult (*)(VkDevice, VkBuffer, VkDeviceMemory, VkDeviceSize);
using PFN_vkMapMemory = VkResult (*)(VkDevice, VkDeviceMemory, VkDeviceSize, VkDeviceSize, VkFlags, void**);
using PFN_vkUnmapMemory = void (*)(VkDevice, VkDeviceMemory);
using PFN_vkCreateShaderModule = VkResult (*)(VkDevice, const VkShaderModuleCreateInfo*, const void*, VkShaderModule*);
using PFN_vkDestroyShaderModule = void (*)(VkDevice, VkShaderModule, const void*);
using PFN_vkCreateDescriptorSetLayout = VkResult (*)(VkDevice, const VkDescriptorSetLayoutCreateInfo*, const void*, VkDescriptorSetLayout*);
using PFN_vkDestroyDescriptorSetLayout = void (*)(VkDevice, VkDescriptorSetLayout, const void*);
using PFN_vkCreatePipelineLayout = VkResult (*)(VkDevice, const VkPipelineLayoutCreateInfo*, const void*, VkPipelineLayout*);
using PFN_vkDestroyPipelineLayout = void (*)(VkDevice, VkPipelineLayout, const void*);
using PFN_vkCreateComputePipelines = VkResult (*)(VkDevice, VkPipelineCache, uint32_t, const VkComputePipelineCreateInfo*, const void*, VkPipeline*);
using PFN_vkDestroyPipeline = void (*)(VkDevice, VkPipeline, const void*);
using PFN_vkCreateDescriptorPool = VkResult (*)(VkDevice, const VkDescriptorPoolCreateInfo*, const void*, VkDescriptorPool*);
using PFN_vkDestroyDescriptorPool = void (*)(VkDevice, VkDescriptorPool, const void*);
using PFN_vkAllocateDescriptorSets = VkResult (*)(VkDevice, const VkDescriptorSetAllocateInfo*, VkDescriptorSet*);
using PFN_vkUpdateDescriptorSets = void (*)(VkDevice, uint32_t, const VkWriteDescriptorSet*, uint32_t, const void*);
using PFN_vkCreateCommandPool = VkResult (*)(VkDevice, const VkCommandPoolCreateInfo*, const void*, VkCommandPool*);
using PFN_vkDestroyCommandPool = void (*)(VkDevice, VkCommandPool, const void*);
using PFN_vkAllocateCommandBuffers = VkResult (*)(VkDevice, const VkCommandBufferAllocateInfo*, VkCommandBuffer*);
using PFN_vkFreeCommandBuffers = void (*)(VkDevice, VkCommandPool, uint32_t, const VkCommandBuffer*);
using PFN_vkBeginCommandBuffer = VkResult (*)(VkCommandBuffer, const VkCommandBufferBeginInfo*);
using PFN_vkEndCommandBuffer = VkResult (*)(VkCommandBuffer);
using PFN_vkCmdBindPipeline = void (*)(VkCommandBuffer, uint32_t, VkPipeline);
using PFN_vkCmdBindDescriptorSets = void (*)(VkCommandBuffer, uint32_t, VkPipelineLayout, uint32_t, uint32_t, const VkDescriptorSet*, uint32_t, const uint32_t*);
using PFN_vkCmdPushConstants = void (*)(VkCommandBuffer, VkPipelineLayout, VkShaderStageFlags, uint32_t, uint32_t, const void*);
using PFN_vkCmdPipelineBarrier = void (*)(VkCommandBuffer, VkPipelineStageFlags, VkPipelineStageFlags, VkFlags, uint32_t, const VkMemoryBarrier*, uint32_t, const void*, uint32_t, const void*);
using PFN_vkCmdCopyBuffer = void (*)(VkCommandBuffer, VkBuffer, VkBuffer, uint32_t, const VkBufferCopy*);
using PFN_vkCmdDispatch = void (*)(VkCommandBuffer, uint32_t, uint32_t, uint32_t);
using PFN_vkCreateFence = VkResult (*)(VkDevice, const VkFenceCreateInfo*, const void*, VkFence*);
using PFN_vkDestroyFence = void (*)(VkDevice, VkFence, const void*);
using PFN_vkWaitForFences = VkResult (*)(VkDevice, uint32_t, const VkFence*, VkBool32, uint64_t);
using PFN_vkQueueSubmit = VkResult (*)(VkQueue, uint32_t, const VkSubmitInfo*, VkFence);
using PFN_vkQueueWaitIdle = VkResult (*)(VkQueue);

} // namespace xbt_vk_abi

// =============================================================================
// Runtime State, Device Probing, Staging Buffers, and Pipeline Cache
// =============================================================================
namespace {

using namespace xbt_vk_abi;

struct VulkanDeviceRecord {
    VkPhysicalDevice phys = nullptr;
    VkPhysicalDeviceProperties props{};
    VkPhysicalDeviceMemoryProperties mem_props{};
    uint32_t compute_queue_family = UINT32_MAX;
    uint32_t subgroup_size = 64u;
    uint32_t tuned_workgroup_size = 64u;
    uint64_t device_local_bytes = 0;
    bool supports_int64 = false;
};

struct VulkanPipelineBundle {
    VkShaderModule shader = 0;
    VkDescriptorSetLayout set_layout = 0;
    VkPipelineLayout pipe_layout = 0;
    VkPipeline pipeline = 0;
};

struct VulkanBuffer {
    VkBuffer buffer = 0;
    VkDeviceMemory memory = 0;
    VkDeviceSize size = 0;
    bool host_visible = false;
};

class VulkanRuntime {
public:
    static VulkanRuntime& instance() {
        static VulkanRuntime rt;
        return rt;
    }

    bool is_available() {
        std::lock_guard<std::mutex> lk(mtx_);
        ensure_initialized_locked();
        return available_;
    }

    int32_t device_count() {
        std::lock_guard<std::mutex> lk(mtx_);
        ensure_initialized_locked();
        return static_cast<int32_t>(devices_.size());
    }

    int32_t get_device_info(
        int32_t dev_idx,
        char* out_name,
        int32_t name_cap,
        uint32_t* out_vendor_id,
        uint32_t* out_device_type,
        uint32_t* out_subgroup_size,
        uint64_t* out_device_local_mem_bytes
    ) {
        std::lock_guard<std::mutex> lk(mtx_);
        ensure_initialized_locked();
        if (!available_ || dev_idx < 0 || static_cast<size_t>(dev_idx) >= devices_.size()) {
            return -1;
        }
        const auto& d = devices_[static_cast<size_t>(dev_idx)];
        if (out_name && name_cap > 0) {
            std::snprintf(out_name, static_cast<size_t>(name_cap), "%s", d.props.deviceName);
        }
        if (out_vendor_id) *out_vendor_id = d.props.vendorID;
        if (out_device_type) *out_device_type = d.props.deviceType;
        if (out_subgroup_size) *out_subgroup_size = d.subgroup_size;
        if (out_device_local_mem_bytes) *out_device_local_mem_bytes = d.device_local_bytes;
        return 0;
    }

    uint32_t active_subgroup_size() {
        std::lock_guard<std::mutex> lk(mtx_);
        ensure_initialized_locked();
        if (!available_ || active_idx_ >= devices_.size()) return 64u;
        return devices_[active_idx_].tuned_workgroup_size;
    }

    void set_spirv_dir(const char* dir) {
        std::lock_guard<std::mutex> lk(mtx_);
        spirv_dir_ = dir ? std::string(dir) : std::string();
    }

    int32_t load_spirv_bytes(int32_t kernel_id, const uint32_t* words, size_t byte_size) {
        std::lock_guard<std::mutex> lk(mtx_);
        if (!words || byte_size == 0 || (byte_size % 4) != 0) return -1;
        if (kernel_id < 0 || kernel_id >= 3) return -2;
        ensure_initialized_locked();
        if (!available_) return -3;

        std::vector<uint32_t> code(words, words + byte_size / sizeof(uint32_t));
        return build_kernel_pipeline_locked(kernel_id, code);
    }

    // Dispatches xbt_lj_dsf_csr.comp
    int32_t dispatch_lj_dsf_csr(
        uint32_t n_atoms,
        uint32_t box_mode,
        uint32_t flags,
        const float* q_f4,
        const float* sig_eps,
        const uint32_t* sym_row_ptr,
        const uint32_t* sym_col_idx,
        uint32_t n_edges,
        const float* sym_wlj,
        const float* sym_wcoul,
        const float* dsf_table_f4,
        const float* box_ubo_32f,
        float rc2,
        float rcut,
        float inv_rc,
        float inv_rc2,
        float du,
        float inv_du,
        float coulomb_prefactor,
        float Arc,
        float Brc,
        float* out_forces_f4,
        float* out_u_half,
        float* out_vir9
    ) {
        std::lock_guard<std::mutex> lk(mtx_);
        ensure_initialized_locked();
        if (!available_) return -10;
        if (n_atoms == 0) return 0;
        if (!q_f4 || !sig_eps || !sym_row_ptr || !dsf_table_f4 || !out_forces_f4 || !out_u_half) return -1;
        if (ensure_kernel_ready_locked(0, "xbt_lj_dsf_csr.spv") != 0) return -20;

        const size_t safe_edges = std::max<size_t>(1, n_edges);
        const bool need_vir = (flags & 8u) != 0u;

        std::vector<VulkanBuffer> bufs(11);
        auto cleanup = [&]() {
            for (auto& b : bufs) destroy_buffer_locked(b);
        };

        auto fail = [&](int32_t code) {
            cleanup();
            return code;
        };

        // Upload read-only buffers (device-local via staging if discrete GPU, or direct if UMA)
        if (create_and_upload_locked(sizeof(float) * 4 * n_atoms, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, q_f4, bufs[0]) != 0) return fail(-30);
        if (create_and_upload_locked(sizeof(float) * 2 * n_atoms, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, sig_eps, bufs[1]) != 0) return fail(-31);
        if (create_and_upload_locked(sizeof(uint32_t) * (n_atoms + 1), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, sym_row_ptr, bufs[2]) != 0) return fail(-32);

        uint32_t dummy_u = 0;
        float dummy_f = 0.0f;
        const void* col_src = (n_edges > 0 && sym_col_idx) ? static_cast<const void*>(sym_col_idx) : &dummy_u;
        const void* wlj_src = (n_edges > 0 && sym_wlj) ? static_cast<const void*>(sym_wlj) : &dummy_f;
        const void* wcoul_src = (n_edges > 0 && sym_wcoul) ? static_cast<const void*>(sym_wcoul) : &dummy_f;

        if (create_and_upload_locked(sizeof(uint32_t) * safe_edges, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, col_src, bufs[3]) != 0) return fail(-33);
        if (create_and_upload_locked(sizeof(float) * safe_edges, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, wlj_src, bufs[4]) != 0) return fail(-34);
        if (create_and_upload_locked(sizeof(float) * safe_edges, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, wcoul_src, bufs[5]) != 0) return fail(-35);
        if (create_and_upload_locked(sizeof(float) * 4 * 4096, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, dsf_table_f4, bufs[6]) != 0) return fail(-36);

        if (create_device_buffer_locked(sizeof(float) * 4 * n_atoms, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, bufs[7]) != 0) return fail(-37);
        if (create_device_buffer_locked(sizeof(float) * n_atoms, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, bufs[8]) != 0) return fail(-38);
        const size_t vir_bytes = sizeof(float) * (need_vir ? 9 * n_atoms : 1);
        if (create_device_buffer_locked(vir_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, bufs[9]) != 0) return fail(-39);

        std::array<float, 32> default_ubo{};
        const float* ubo_ptr = box_ubo_32f ? box_ubo_32f : default_ubo.data();
        if (create_and_upload_locked(sizeof(float) * 32, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, ubo_ptr, bufs[10]) != 0) return fail(-40);

        struct PushConsts {
            uint32_t n_atoms;
            uint32_t box_mode;
            uint32_t flags;
            uint32_t pad0;
            float rc2;
            float rcut;
            float inv_rc;
            float inv_rc2;
            float du;
            float inv_du;
            float coulomb_prefactor;
            float Arc;
            float Brc;
        } pc{n_atoms, box_mode, flags, 0u, rc2, rcut, inv_rc, inv_rc2, du, inv_du, coulomb_prefactor, Arc, Brc};

        const uint32_t wg = devices_[active_idx_].tuned_workgroup_size;
        const uint32_t gx = (n_atoms + wg - 1u) / wg;
        std::array<uint32_t, 11> types{};
        types.fill(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        types[10] = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;

        if (execute_compute_locked(kernels_[0], bufs.data(), types.data(), 11, &pc, sizeof(pc), gx, 1u, 1u) != 0) {
            return fail(-50);
        }

        if (readback_buffer_locked(bufs[7], sizeof(float) * 4 * n_atoms, out_forces_f4) != 0) return fail(-60);
        if (readback_buffer_locked(bufs[8], sizeof(float) * n_atoms, out_u_half) != 0) return fail(-61);
        if (need_vir && out_vir9) {
            if (readback_buffer_locked(bufs[9], sizeof(float) * 9 * n_atoms, out_vir9) != 0) return fail(-62);
        }

        cleanup();
        return 0;
    }

    int32_t dispatch_ternary_matvec(
        uint32_t M,
        uint32_t K_words,
        const uint64_t* W_P,
        const uint64_t* W_N,
        const uint64_t* x_P,
        const uint64_t* x_N,
        const float* row_scales,
        float x_scale,
        float* y_out
    ) {
        std::lock_guard<std::mutex> lk(mtx_);
        ensure_initialized_locked();
        if (!available_) return -10;
        if (M == 0) return 0;
        if (!W_P || !W_N || !x_P || !x_N || !y_out) return -1;
        if (ensure_kernel_ready_locked(1, "xbt_ternary_matvec.spv") != 0) return -20;

        const size_t w_words = std::max<size_t>(1, size_t(M) * size_t(K_words));
        const size_t k_safe = std::max<size_t>(1, K_words);
        std::vector<VulkanBuffer> bufs(6);
        auto cleanup = [&]() {
            for (auto& b : bufs) destroy_buffer_locked(b);
        };
        auto fail = [&](int32_t c) { cleanup(); return c; };

        if (create_and_upload_locked(sizeof(uint64_t) * w_words, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, W_P, bufs[0]) != 0) return fail(-30);
        if (create_and_upload_locked(sizeof(uint64_t) * w_words, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, W_N, bufs[1]) != 0) return fail(-31);
        if (create_and_upload_locked(sizeof(uint64_t) * k_safe, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, x_P, bufs[2]) != 0) return fail(-32);
        if (create_and_upload_locked(sizeof(uint64_t) * k_safe, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, x_N, bufs[3]) != 0) return fail(-33);

        std::vector<float> unit_scales;
        const float* rs_ptr = row_scales;
        if (!rs_ptr) {
            unit_scales.assign(M, 1.0f);
            rs_ptr = unit_scales.data();
        }
        if (create_and_upload_locked(sizeof(float) * M, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, rs_ptr, bufs[4]) != 0) return fail(-34);
        if (create_device_buffer_locked(sizeof(float) * M, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, bufs[5]) != 0) return fail(-35);

        struct PushConsts {
            uint32_t M;
            uint32_t K_words;
            float x_scale;
            uint32_t has_row_scales;
        } pc{M, K_words, x_scale, row_scales ? 1u : 0u};

        const uint32_t wg = devices_[active_idx_].tuned_workgroup_size;
        const uint32_t gx = (M + wg - 1u) / wg;
        std::array<uint32_t, 6> types{};
        types.fill(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);

        if (execute_compute_locked(kernels_[1], bufs.data(), types.data(), 6, &pc, sizeof(pc), gx, 1u, 1u) != 0) {
            return fail(-50);
        }
        if (readback_buffer_locked(bufs[5], sizeof(float) * M, y_out) != 0) return fail(-60);

        cleanup();
        return 0;
    }

    int32_t dispatch_msm_stencil27(
        uint32_t Ml,
        const float* q_in,
        const float* w27,
        uint32_t accumulate,
        float* phi_out
    ) {
        std::lock_guard<std::mutex> lk(mtx_);
        ensure_initialized_locked();
        if (!available_) return -10;
        if (Ml == 0) return 0;
        if (!q_in || !w27 || !phi_out) return -1;
        if (ensure_kernel_ready_locked(2, "xbt_msm_stencil27.spv") != 0) return -20;

        const size_t n_grid = size_t(Ml) * size_t(Ml) * size_t(Ml);
        std::vector<VulkanBuffer> bufs(3);
        auto cleanup = [&]() {
            for (auto& b : bufs) destroy_buffer_locked(b);
        };
        auto fail = [&](int32_t c) { cleanup(); return c; };

        if (create_and_upload_locked(sizeof(float) * n_grid, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, q_in, bufs[0]) != 0) return fail(-30);
        if (create_and_upload_locked(sizeof(float) * 27, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, w27, bufs[1]) != 0) return fail(-31);
        if (accumulate != 0u) {
            if (create_and_upload_locked(sizeof(float) * n_grid, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, phi_out, bufs[2]) != 0) return fail(-32);
        } else {
            if (create_device_buffer_locked(sizeof(float) * n_grid, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, bufs[2]) != 0) return fail(-32);
        }

        struct PushConsts {
            uint32_t Ml;
            uint32_t accumulate;
        } pc{Ml, accumulate};

        const uint32_t g = (Ml + 3u) / 4u;
        std::array<uint32_t, 3> types{};
        types.fill(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);

        if (execute_compute_locked(kernels_[2], bufs.data(), types.data(), 3, &pc, sizeof(pc), g, g, g) != 0) {
            return fail(-50);
        }
        if (readback_buffer_locked(bufs[2], sizeof(float) * n_grid, phi_out) != 0) return fail(-60);

        cleanup();
        return 0;
    }

private:
    VulkanRuntime() = default;

    ~VulkanRuntime() {
        if (device_) {
            for (auto& k : kernels_) {
                if (k.pipeline && vkDestroyPipeline) vkDestroyPipeline(device_, k.pipeline, nullptr);
                if (k.pipe_layout && vkDestroyPipelineLayout) vkDestroyPipelineLayout(device_, k.pipe_layout, nullptr);
                if (k.set_layout && vkDestroyDescriptorSetLayout) vkDestroyDescriptorSetLayout(device_, k.set_layout, nullptr);
                if (k.shader && vkDestroyShaderModule) vkDestroyShaderModule(device_, k.shader, nullptr);
            }
            if (cmd_pool_ && vkDestroyCommandPool) vkDestroyCommandPool(device_, cmd_pool_, nullptr);
            if (desc_pool_ && vkDestroyDescriptorPool) vkDestroyDescriptorPool(device_, desc_pool_, nullptr);
            if (vkDestroyDevice) vkDestroyDevice(device_, nullptr);
        }
        if (vk_instance_ && vkDestroyInstance) {
            vkDestroyInstance(vk_instance_, nullptr);
        }
        if (lib_handle_) {
            vk_dlclose(lib_handle_);
        }
    }

    template <typename Fn>
    bool load_inst_fn(const char* name, Fn& out) {
        if (!vkGetInstanceProcAddr) return false;
        out = reinterpret_cast<Fn>(vkGetInstanceProcAddr(vk_instance_, name));
        return out != nullptr;
    }

    void ensure_initialized_locked() {
        if (init_attempted_) return;
        init_attempted_ = true;

#if defined(_WIN32)
        const char* candidates[] = {
            "vulkan-1.dll"
        };
#elif defined(__APPLE__)
        const char* candidates[] = {
            "libvulkan.1.dylib",
            "libvulkan.dylib",
            "libMoltenVK.dylib",
            "/opt/homebrew/lib/libvulkan.1.dylib",
            "/opt/homebrew/lib/libMoltenVK.dylib",
            "/usr/local/lib/libvulkan.1.dylib",
            "/usr/local/lib/libMoltenVK.dylib"
        };
#else
        const char* candidates[] = {
            "libvulkan.so.1",
            "libvulkan.so"
        };
#endif
        for (const char* path : candidates) {
            lib_handle_ = vk_dlopen(path);
            if (lib_handle_) break;
        }
        if (!lib_handle_) return;

        vkGetInstanceProcAddr = reinterpret_cast<PFN_vkGetInstanceProcAddr>(vk_dlsym(lib_handle_, "vkGetInstanceProcAddr"));
        if (!vkGetInstanceProcAddr) return;

        vkCreateInstance = reinterpret_cast<PFN_vkCreateInstance>(vkGetInstanceProcAddr(nullptr, "vkCreateInstance"));
        if (!vkCreateInstance) return;

        VkApplicationInfo app_info{};
        app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app_info.pApplicationName = "TriumXBT";
        app_info.applicationVersion = 1;
        app_info.pEngineName = "TriumPortableGPU";
        app_info.engineVersion = 1;
        app_info.apiVersion = VK_API_VERSION_1_1;

        const char* port_exts[] = {"VK_KHR_portability_enumeration"};
        VkInstanceCreateInfo inst_ci{};
        inst_ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        inst_ci.pApplicationInfo = &app_info;
#if defined(__APPLE__)
        inst_ci.flags = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
        inst_ci.enabledExtensionCount = 1;
        inst_ci.ppEnabledExtensionNames = port_exts;
#endif

        if (vkCreateInstance(&inst_ci, nullptr, &vk_instance_) != VK_SUCCESS) {
            inst_ci.flags = 0;
            inst_ci.enabledExtensionCount = 0;
            inst_ci.ppEnabledExtensionNames = nullptr;
            if (vkCreateInstance(&inst_ci, nullptr, &vk_instance_) != VK_SUCCESS) {
                vk_instance_ = nullptr;
                return;
            }
        }

        if (!load_inst_fn("vkDestroyInstance", vkDestroyInstance) ||
            !load_inst_fn("vkEnumeratePhysicalDevices", vkEnumeratePhysicalDevices) ||
            !load_inst_fn("vkGetPhysicalDeviceProperties", vkGetPhysicalDeviceProperties) ||
            !load_inst_fn("vkGetPhysicalDeviceMemoryProperties", vkGetPhysicalDeviceMemoryProperties) ||
            !load_inst_fn("vkGetPhysicalDeviceQueueFamilyProperties", vkGetPhysicalDeviceQueueFamilyProperties) ||
            !load_inst_fn("vkCreateDevice", vkCreateDevice) ||
            !load_inst_fn("vkDestroyDevice", vkDestroyDevice) ||
            !load_inst_fn("vkGetDeviceQueue", vkGetDeviceQueue) ||
            !load_inst_fn("vkCreateBuffer", vkCreateBuffer) ||
            !load_inst_fn("vkDestroyBuffer", vkDestroyBuffer) ||
            !load_inst_fn("vkGetBufferMemoryRequirements", vkGetBufferMemoryRequirements) ||
            !load_inst_fn("vkAllocateMemory", vkAllocateMemory) ||
            !load_inst_fn("vkFreeMemory", vkFreeMemory) ||
            !load_inst_fn("vkBindBufferMemory", vkBindBufferMemory) ||
            !load_inst_fn("vkMapMemory", vkMapMemory) ||
            !load_inst_fn("vkUnmapMemory", vkUnmapMemory) ||
            !load_inst_fn("vkCreateShaderModule", vkCreateShaderModule) ||
            !load_inst_fn("vkDestroyShaderModule", vkDestroyShaderModule) ||
            !load_inst_fn("vkCreateDescriptorSetLayout", vkCreateDescriptorSetLayout) ||
            !load_inst_fn("vkDestroyDescriptorSetLayout", vkDestroyDescriptorSetLayout) ||
            !load_inst_fn("vkCreatePipelineLayout", vkCreatePipelineLayout) ||
            !load_inst_fn("vkDestroyPipelineLayout", vkDestroyPipelineLayout) ||
            !load_inst_fn("vkCreateComputePipelines", vkCreateComputePipelines) ||
            !load_inst_fn("vkDestroyPipeline", vkDestroyPipeline) ||
            !load_inst_fn("vkCreateDescriptorPool", vkCreateDescriptorPool) ||
            !load_inst_fn("vkDestroyDescriptorPool", vkDestroyDescriptorPool) ||
            !load_inst_fn("vkAllocateDescriptorSets", vkAllocateDescriptorSets) ||
            !load_inst_fn("vkUpdateDescriptorSets", vkUpdateDescriptorSets) ||
            !load_inst_fn("vkCreateCommandPool", vkCreateCommandPool) ||
            !load_inst_fn("vkDestroyCommandPool", vkDestroyCommandPool) ||
            !load_inst_fn("vkAllocateCommandBuffers", vkAllocateCommandBuffers) ||
            !load_inst_fn("vkFreeCommandBuffers", vkFreeCommandBuffers) ||
            !load_inst_fn("vkBeginCommandBuffer", vkBeginCommandBuffer) ||
            !load_inst_fn("vkEndCommandBuffer", vkEndCommandBuffer) ||
            !load_inst_fn("vkCmdBindPipeline", vkCmdBindPipeline) ||
            !load_inst_fn("vkCmdBindDescriptorSets", vkCmdBindDescriptorSets) ||
            !load_inst_fn("vkCmdPushConstants", vkCmdPushConstants) ||
            !load_inst_fn("vkCmdPipelineBarrier", vkCmdPipelineBarrier) ||
            !load_inst_fn("vkCmdCopyBuffer", vkCmdCopyBuffer) ||
            !load_inst_fn("vkCmdDispatch", vkCmdDispatch) ||
            !load_inst_fn("vkCreateFence", vkCreateFence) ||
            !load_inst_fn("vkDestroyFence", vkDestroyFence) ||
            !load_inst_fn("vkWaitForFences", vkWaitForFences) ||
            !load_inst_fn("vkQueueSubmit", vkQueueSubmit) ||
            !load_inst_fn("vkQueueWaitIdle", vkQueueWaitIdle)) {
            return;
        }
        load_inst_fn("vkGetPhysicalDeviceProperties2", vkGetPhysicalDeviceProperties2);
        if (!vkGetPhysicalDeviceProperties2) {
            load_inst_fn("vkGetPhysicalDeviceProperties2KHR", vkGetPhysicalDeviceProperties2);
        }

        uint32_t count = 0;
        if (vkEnumeratePhysicalDevices(vk_instance_, &count, nullptr) != VK_SUCCESS || count == 0) {
            return;
        }
        std::vector<VkPhysicalDevice> phys_list(count);
        VkResult er = vkEnumeratePhysicalDevices(vk_instance_, &count, phys_list.data());
        if (er != VK_SUCCESS && er != VK_INCOMPLETE) return;

        for (uint32_t idx = 0; idx < count; ++idx) {
            VkPhysicalDevice pd = phys_list[idx];
            VulkanDeviceRecord rec{};
            rec.phys = pd;

            vkGetPhysicalDeviceProperties(pd, &rec.props);
            vkGetPhysicalDeviceMemoryProperties(pd, &rec.mem_props);

            uint32_t qf_count = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(pd, &qf_count, nullptr);
            std::vector<VkQueueFamilyProperties> qfs(qf_count);
            if (qf_count > 0) {
                vkGetPhysicalDeviceQueueFamilyProperties(pd, &qf_count, qfs.data());
            }
            for (uint32_t qf = 0; qf < qf_count; ++qf) {
                if ((qfs[qf].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0u) {
                    rec.compute_queue_family = qf;
                    break;
                }
            }
            if (rec.compute_queue_family == UINT32_MAX) continue;

            // Query VkPhysicalDeviceSubgroupProperties when Vulkan 1.1+ is available
            rec.subgroup_size = (rec.props.vendorID == 0x10DE || rec.props.vendorID == 0x106B) ? 32u : 64u;
            if (vkGetPhysicalDeviceProperties2 && rec.props.apiVersion >= VK_API_VERSION_1_1) {
                VkPhysicalDeviceSubgroupProperties sg{};
                sg.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
                VkPhysicalDeviceProperties2 props2{};
                props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
                props2.pNext = &sg;
                vkGetPhysicalDeviceProperties2(pd, &props2);
                if (sg.subgroupSize >= 8u && sg.subgroupSize <= 128u) {
                    rec.subgroup_size = sg.subgroupSize;
                }
            }
            rec.tuned_workgroup_size = (rec.subgroup_size <= 32u) ? 32u : 64u;

            uint64_t dev_mem = 0;
            for (uint32_t h = 0; h < rec.mem_props.memoryHeapCount; ++h) {
                if ((rec.mem_props.memoryHeaps[h].flags & 1u) != 0u) {
                    dev_mem += rec.mem_props.memoryHeaps[h].size;
                }
            }
            rec.device_local_bytes = dev_mem;
            devices_.push_back(rec);
        }

        if (devices_.empty()) return;

        auto score = [](uint32_t dev_type) -> int {
            switch (dev_type) {
                case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: return 4;
                case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return 3;
                case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: return 2;
                case VK_PHYSICAL_DEVICE_TYPE_CPU: return 1;
                default: return 0;
            }
        };
        active_idx_ = 0;
        for (size_t i = 1; i < devices_.size(); ++i) {
            if (score(devices_[i].props.deviceType) > score(devices_[active_idx_].props.deviceType)) {
                active_idx_ = i;
            }
        }

        const auto& active = devices_[active_idx_];
        float prio = 1.0f;
        VkDeviceQueueCreateInfo qci{};
        qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qci.queueFamilyIndex = active.compute_queue_family;
        qci.queueCount = 1;
        qci.pQueuePriorities = &prio;

        VkDeviceCreateInfo dci{};
        dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos = &qci;

        if (vkCreateDevice(active.phys, &dci, nullptr, &device_) != VK_SUCCESS) {
            device_ = nullptr;
            return;
        }
        vkGetDeviceQueue(device_, active.compute_queue_family, 0, &queue_);

        VkCommandPoolCreateInfo cpci{};
        cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        cpci.queueFamilyIndex = active.compute_queue_family;
        if (vkCreateCommandPool(device_, &cpci, nullptr, &cmd_pool_) != VK_SUCCESS) return;

        VkDescriptorPoolSize pool_sizes[2] = {
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 256u},
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 64u}
        };
        VkDescriptorPoolCreateInfo dpci{};
        dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        dpci.maxSets = 64u;
        dpci.poolSizeCount = 2u;
        dpci.pPoolSizes = pool_sizes;
        if (vkCreateDescriptorPool(device_, &dpci, nullptr, &desc_pool_) != VK_SUCCESS) return;

        available_ = true;
    }

    uint32_t find_memory_type_locked(uint32_t type_filter, VkMemoryPropertyFlags required) const {
        const auto& mp = devices_[active_idx_].mem_props;
        for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
            if ((type_filter & (1u << i)) != 0u &&
                (mp.memoryTypes[i].propertyFlags & required) == required) {
                return i;
            }
        }
        return UINT32_MAX;
    }

    int32_t create_buffer_raw_locked(
        VkDeviceSize size,
        VkBufferUsageFlags usage,
        VkMemoryPropertyFlags mem_props,
        VulkanBuffer& out
    ) {
        VkBufferCreateInfo bci{};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size = size;
        bci.usage = usage;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateBuffer(device_, &bci, nullptr, &out.buffer) != VK_SUCCESS) return -1;

        VkMemoryRequirements req{};
        vkGetBufferMemoryRequirements(device_, out.buffer, &req);

        uint32_t type_idx = find_memory_type_locked(req.memoryTypeBits, mem_props);
        if (type_idx == UINT32_MAX) {
            vkDestroyBuffer(device_, out.buffer, nullptr);
            out.buffer = 0;
            return -2;
        }

        VkMemoryAllocateInfo mai{};
        mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.allocationSize = req.size;
        mai.memoryTypeIndex = type_idx;
        if (vkAllocateMemory(device_, &mai, nullptr, &out.memory) != VK_SUCCESS) {
            vkDestroyBuffer(device_, out.buffer, nullptr);
            out.buffer = 0;
            return -3;
        }

        if (vkBindBufferMemory(device_, out.buffer, out.memory, 0) != VK_SUCCESS) {
            vkFreeMemory(device_, out.memory, nullptr);
            vkDestroyBuffer(device_, out.buffer, nullptr);
            out.buffer = 0;
            out.memory = 0;
            return -4;
        }

        out.size = size;
        out.host_visible = (mem_props & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0u;
        return 0;
    }

    void destroy_buffer_locked(VulkanBuffer& buf) {
        if (buf.buffer && vkDestroyBuffer) vkDestroyBuffer(device_, buf.buffer, nullptr);
        if (buf.memory && vkFreeMemory) vkFreeMemory(device_, buf.memory, nullptr);
        buf = VulkanBuffer{};
    }

    int32_t create_device_buffer_locked(VkDeviceSize size, VkBufferUsageFlags usage, VulkanBuffer& out) {
        // Prefer UMA device-local + host-visible if available, else pure device-local
        if (create_buffer_raw_locked(
                size, usage,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                out) == 0) {
            return 0;
        }
        return create_buffer_raw_locked(size, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, out);
    }

    int32_t copy_buffer_sync_locked(VkBuffer src, VkBuffer dst, VkDeviceSize size) {
        VkCommandBufferAllocateInfo cbai{};
        cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cbai.commandPool = cmd_pool_;
        cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbai.commandBufferCount = 1u;

        VkCommandBuffer cmd = nullptr;
        if (vkAllocateCommandBuffers(device_, &cbai, &cmd) != VK_SUCCESS) return -1;

        VkCommandBufferBeginInfo cbbi{};
        cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd, &cbbi);

        VkBufferCopy region{0, 0, size};
        vkCmdCopyBuffer(cmd, src, dst, 1u, &region);
        vkEndCommandBuffer(cmd);

        VkSubmitInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1u;
        si.pCommandBuffers = &cmd;

        VkFenceCreateInfo fci{};
        fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        VkFence fence = 0;
        vkCreateFence(device_, &fci, nullptr, &fence);
        vkQueueSubmit(queue_, 1u, &si, fence);
        vkWaitForFences(device_, 1u, &fence, 1u, ~uint64_t(0));
        vkDestroyFence(device_, fence, nullptr);
        vkFreeCommandBuffers(device_, cmd_pool_, 1u, &cmd);
        return 0;
    }

    int32_t create_and_upload_locked(
        VkDeviceSize size,
        VkBufferUsageFlags usage,
        const void* src_data,
        VulkanBuffer& out
    ) {
        // Fast path: unified memory (device-local + host-visible)
        if (create_buffer_raw_locked(
                size, usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                out) == 0) {
            void* mapped = nullptr;
            if (vkMapMemory(device_, out.memory, 0, size, 0, &mapped) != VK_SUCCESS) return -1;
            std::memcpy(mapped, src_data, static_cast<size_t>(size));
            vkUnmapMemory(device_, out.memory);
            return 0;
        }

        // Discrete GPU staging path: host-visible staging -> device-local buffer
        VulkanBuffer staging{};
        if (create_buffer_raw_locked(
                size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                staging) != 0) {
            return -2;
        }
        void* mapped = nullptr;
        if (vkMapMemory(device_, staging.memory, 0, size, 0, &mapped) != VK_SUCCESS) {
            destroy_buffer_locked(staging);
            return -3;
        }
        std::memcpy(mapped, src_data, static_cast<size_t>(size));
        vkUnmapMemory(device_, staging.memory);

        if (create_buffer_raw_locked(size, usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, out) != 0) {
            destroy_buffer_locked(staging);
            return -4;
        }
        int32_t cr = copy_buffer_sync_locked(staging.buffer, out.buffer, size);
        destroy_buffer_locked(staging);
        return cr;
    }

    int32_t readback_buffer_locked(const VulkanBuffer& src, VkDeviceSize size, void* dst_data) {
        if (src.host_visible) {
            void* mapped = nullptr;
            if (vkMapMemory(device_, src.memory, 0, size, 0, &mapped) != VK_SUCCESS) return -1;
            std::memcpy(dst_data, mapped, static_cast<size_t>(size));
            vkUnmapMemory(device_, src.memory);
            return 0;
        }

        VulkanBuffer staging{};
        if (create_buffer_raw_locked(
                size, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                staging) != 0) {
            return -2;
        }
        if (copy_buffer_sync_locked(src.buffer, staging.buffer, size) != 0) {
            destroy_buffer_locked(staging);
            return -3;
        }
        void* mapped = nullptr;
        if (vkMapMemory(device_, staging.memory, 0, size, 0, &mapped) != VK_SUCCESS) {
            destroy_buffer_locked(staging);
            return -4;
        }
        std::memcpy(dst_data, mapped, static_cast<size_t>(size));
        vkUnmapMemory(device_, staging.memory);
        destroy_buffer_locked(staging);
        return 0;
    }

    int32_t ensure_kernel_ready_locked(int32_t kernel_id, const char* filename) {
        if (kernels_[kernel_id].pipeline != 0) return 0;
        std::string path = spirv_dir_.empty() ? std::string(filename) : (spirv_dir_ + "/" + filename);
        std::ifstream ifs(path, std::ios::binary | std::ios::ate);
        if (!ifs.is_open()) return -1;
        std::streamsize sz = ifs.tellg();
        if (sz <= 0 || (sz % 4) != 0) return -2;
        ifs.seekg(0, std::ios::beg);
        std::vector<uint32_t> code(static_cast<size_t>(sz) / sizeof(uint32_t));
        if (!ifs.read(reinterpret_cast<char*>(code.data()), sz)) return -3;
        return build_kernel_pipeline_locked(kernel_id, code);
    }

    int32_t build_kernel_pipeline_locked(int32_t kernel_id, const std::vector<uint32_t>& code) {
        auto& k = kernels_[kernel_id];
        if (k.pipeline) vkDestroyPipeline(device_, k.pipeline, nullptr);
        if (k.pipe_layout) vkDestroyPipelineLayout(device_, k.pipe_layout, nullptr);
        if (k.set_layout) vkDestroyDescriptorSetLayout(device_, k.set_layout, nullptr);
        if (k.shader) vkDestroyShaderModule(device_, k.shader, nullptr);
        k = VulkanPipelineBundle{};

        VkShaderModuleCreateInfo smci{};
        smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smci.codeSize = code.size() * sizeof(uint32_t);
        smci.pCode = code.data();
        if (vkCreateShaderModule(device_, &smci, nullptr, &k.shader) != VK_SUCCESS) return -1;

        uint32_t n_bindings = 0;
        uint32_t pc_bytes = 0;
        std::vector<VkDescriptorSetLayoutBinding> bindings;
        if (kernel_id == 0) {
            n_bindings = 11u;
            pc_bytes = 52u;
            bindings.resize(n_bindings);
            for (uint32_t b = 0; b < n_bindings; ++b) {
                bindings[b].binding = b;
                bindings[b].descriptorType = (b == 10u) ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                bindings[b].descriptorCount = 1u;
                bindings[b].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            }
        } else if (kernel_id == 1) {
            n_bindings = 6u;
            pc_bytes = 16u;
            bindings.resize(n_bindings);
            for (uint32_t b = 0; b < n_bindings; ++b) {
                bindings[b].binding = b;
                bindings[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                bindings[b].descriptorCount = 1u;
                bindings[b].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            }
        } else {
            n_bindings = 3u;
            pc_bytes = 8u;
            bindings.resize(n_bindings);
            for (uint32_t b = 0; b < n_bindings; ++b) {
                bindings[b].binding = b;
                bindings[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                bindings[b].descriptorCount = 1u;
                bindings[b].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            }
        }

        VkDescriptorSetLayoutCreateInfo dslci{};
        dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dslci.bindingCount = n_bindings;
        dslci.pBindings = bindings.data();
        if (vkCreateDescriptorSetLayout(device_, &dslci, nullptr, &k.set_layout) != VK_SUCCESS) return -2;

        VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0u, pc_bytes};
        VkPipelineLayoutCreateInfo plci{};
        plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount = 1u;
        plci.pSetLayouts = &k.set_layout;
        plci.pushConstantRangeCount = 1u;
        plci.pPushConstantRanges = &pcr;
        if (vkCreatePipelineLayout(device_, &plci, nullptr, &k.pipe_layout) != VK_SUCCESS) return -3;

        // Specialize adaptive WORKGROUP_SIZE (constant_id 0 & 1) from vendor subgroupSize
        uint32_t spec_wg = (kernel_id == 2) ? 4u : devices_[active_idx_].tuned_workgroup_size;
        VkSpecializationMapEntry map_entries[2] = {
            {0u, 0u, sizeof(uint32_t)},
            {1u, 0u, sizeof(uint32_t)}
        };
        VkSpecializationInfo spec_info{};
        spec_info.mapEntryCount = (kernel_id == 2) ? 1u : 2u;
        spec_info.pMapEntries = map_entries;
        spec_info.dataSize = sizeof(uint32_t);
        spec_info.pData = &spec_wg;

        VkComputePipelineCreateInfo cpci{};
        cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cpci.stage.module = k.shader;
        cpci.stage.pName = "main";
        cpci.stage.pSpecializationInfo = &spec_info;
        cpci.layout = k.pipe_layout;

        if (vkCreateComputePipelines(device_, 0, 1u, &cpci, nullptr, &k.pipeline) != VK_SUCCESS) return -4;
        return 0;
    }

    int32_t execute_compute_locked(
        const VulkanPipelineBundle& k,
        const VulkanBuffer* bufs,
        const uint32_t* desc_types,
        uint32_t n_bufs,
        const void* push_data,
        uint32_t push_bytes,
        uint32_t gx,
        uint32_t gy,
        uint32_t gz
    ) {
        VkDescriptorSetAllocateInfo dsai{};
        dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dsai.descriptorPool = desc_pool_;
        dsai.descriptorSetCount = 1u;
        dsai.pSetLayouts = &k.set_layout;
        VkDescriptorSet dset = 0;
        if (vkAllocateDescriptorSets(device_, &dsai, &dset) != VK_SUCCESS) return -1;

        std::vector<VkDescriptorBufferInfo> dbis(n_bufs);
        std::vector<VkWriteDescriptorSet> writes(n_bufs);
        for (uint32_t i = 0; i < n_bufs; ++i) {
            dbis[i] = VkDescriptorBufferInfo{bufs[i].buffer, 0, VK_WHOLE_SIZE};
            writes[i] = VkWriteDescriptorSet{};
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = dset;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1u;
            writes[i].descriptorType = desc_types[i];
            writes[i].pBufferInfo = &dbis[i];
        }
        vkUpdateDescriptorSets(device_, n_bufs, writes.data(), 0u, nullptr);

        VkCommandBufferAllocateInfo cbai{};
        cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cbai.commandPool = cmd_pool_;
        cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbai.commandBufferCount = 1u;
        VkCommandBuffer cmd = nullptr;
        if (vkAllocateCommandBuffers(device_, &cbai, &cmd) != VK_SUCCESS) return -2;

        VkCommandBufferBeginInfo cbbi{};
        cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd, &cbbi);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, k.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, k.pipe_layout, 0u, 1u, &dset, 0u, nullptr);
        vkCmdPushConstants(cmd, k.pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0u, push_bytes, push_data);
        vkCmdDispatch(cmd, gx, gy, gz);

        VkMemoryBarrier mbar{};
        mbar.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        mbar.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mbar.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0u, 1u, &mbar, 0u, nullptr, 0u, nullptr);
        vkEndCommandBuffer(cmd);

        VkSubmitInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1u;
        si.pCommandBuffers = &cmd;

        VkFenceCreateInfo fci{};
        fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        VkFence fence = 0;
        vkCreateFence(device_, &fci, nullptr, &fence);
        VkResult sr = vkQueueSubmit(queue_, 1u, &si, fence);
        if (sr == VK_SUCCESS) {
            vkWaitForFences(device_, 1u, &fence, 1u, ~uint64_t(0));
        }
        vkDestroyFence(device_, fence, nullptr);
        vkFreeCommandBuffers(device_, cmd_pool_, 1u, &cmd);
        return (sr == VK_SUCCESS) ? 0 : -3;
    }

    std::mutex mtx_;
    bool init_attempted_ = false;
    bool available_ = false;
    void* lib_handle_ = nullptr;
    VkInstance vk_instance_ = nullptr;
    VkDevice device_ = nullptr;
    VkQueue queue_ = nullptr;
    VkCommandPool cmd_pool_ = 0;
    VkDescriptorPool desc_pool_ = 0;
    std::vector<VulkanDeviceRecord> devices_;
    size_t active_idx_ = 0;
    std::string spirv_dir_;
    std::array<VulkanPipelineBundle, 3> kernels_{};

    PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr = nullptr;
    PFN_vkCreateInstance vkCreateInstance = nullptr;
    PFN_vkDestroyInstance vkDestroyInstance = nullptr;
    PFN_vkEnumeratePhysicalDevices vkEnumeratePhysicalDevices = nullptr;
    PFN_vkGetPhysicalDeviceProperties vkGetPhysicalDeviceProperties = nullptr;
    PFN_vkGetPhysicalDeviceProperties2 vkGetPhysicalDeviceProperties2 = nullptr;
    PFN_vkGetPhysicalDeviceMemoryProperties vkGetPhysicalDeviceMemoryProperties = nullptr;
    PFN_vkGetPhysicalDeviceQueueFamilyProperties vkGetPhysicalDeviceQueueFamilyProperties = nullptr;
    PFN_vkCreateDevice vkCreateDevice = nullptr;
    PFN_vkDestroyDevice vkDestroyDevice = nullptr;
    PFN_vkGetDeviceQueue vkGetDeviceQueue = nullptr;
    PFN_vkCreateBuffer vkCreateBuffer = nullptr;
    PFN_vkDestroyBuffer vkDestroyBuffer = nullptr;
    PFN_vkGetBufferMemoryRequirements vkGetBufferMemoryRequirements = nullptr;
    PFN_vkAllocateMemory vkAllocateMemory = nullptr;
    PFN_vkFreeMemory vkFreeMemory = nullptr;
    PFN_vkBindBufferMemory vkBindBufferMemory = nullptr;
    PFN_vkMapMemory vkMapMemory = nullptr;
    PFN_vkUnmapMemory vkUnmapMemory = nullptr;
    PFN_vkCreateShaderModule vkCreateShaderModule = nullptr;
    PFN_vkDestroyShaderModule vkDestroyShaderModule = nullptr;
    PFN_vkCreateDescriptorSetLayout vkCreateDescriptorSetLayout = nullptr;
    PFN_vkDestroyDescriptorSetLayout vkDestroyDescriptorSetLayout = nullptr;
    PFN_vkCreatePipelineLayout vkCreatePipelineLayout = nullptr;
    PFN_vkDestroyPipelineLayout vkDestroyPipelineLayout = nullptr;
    PFN_vkCreateComputePipelines vkCreateComputePipelines = nullptr;
    PFN_vkDestroyPipeline vkDestroyPipeline = nullptr;
    PFN_vkCreateDescriptorPool vkCreateDescriptorPool = nullptr;
    PFN_vkDestroyDescriptorPool vkDestroyDescriptorPool = nullptr;
    PFN_vkAllocateDescriptorSets vkAllocateDescriptorSets = nullptr;
    PFN_vkUpdateDescriptorSets vkUpdateDescriptorSets = nullptr;
    PFN_vkCreateCommandPool vkCreateCommandPool = nullptr;
    PFN_vkDestroyCommandPool vkDestroyCommandPool = nullptr;
    PFN_vkAllocateCommandBuffers vkAllocateCommandBuffers = nullptr;
    PFN_vkFreeCommandBuffers vkFreeCommandBuffers = nullptr;
    PFN_vkBeginCommandBuffer vkBeginCommandBuffer = nullptr;
    PFN_vkEndCommandBuffer vkEndCommandBuffer = nullptr;
    PFN_vkCmdBindPipeline vkCmdBindPipeline = nullptr;
    PFN_vkCmdBindDescriptorSets vkCmdBindDescriptorSets = nullptr;
    PFN_vkCmdPushConstants vkCmdPushConstants = nullptr;
    PFN_vkCmdPipelineBarrier vkCmdPipelineBarrier = nullptr;
    PFN_vkCmdCopyBuffer vkCmdCopyBuffer = nullptr;
    PFN_vkCmdDispatch vkCmdDispatch = nullptr;
    PFN_vkCreateFence vkCreateFence = nullptr;
    PFN_vkDestroyFence vkDestroyFence = nullptr;
    PFN_vkWaitForFences vkWaitForFences = nullptr;
    PFN_vkQueueSubmit vkQueueSubmit = nullptr;
    PFN_vkQueueWaitIdle vkQueueWaitIdle = nullptr;
};

} // anonymous namespace

// =============================================================================
// Exported C ABI Entrypoints
// =============================================================================

XBT_EXPORT int32_t xbt_vulkan_available(void) {
    return VulkanRuntime::instance().is_available() ? 1 : 0;
}

XBT_EXPORT int32_t xbt_vulkan_device_count(void) {
    return VulkanRuntime::instance().device_count();
}

XBT_EXPORT int32_t xbt_vulkan_device_info(
    int32_t dev_idx,
    char* out_name,
    int32_t name_cap,
    uint32_t* out_vendor_id,
    uint32_t* out_device_type,
    uint32_t* out_subgroup_size,
    uint64_t* out_device_local_mem_bytes
) {
    return VulkanRuntime::instance().get_device_info(
        dev_idx, out_name, name_cap, out_vendor_id, out_device_type,
        out_subgroup_size, out_device_local_mem_bytes);
}

XBT_EXPORT uint32_t xbt_vulkan_subgroup_size(void) {
    return VulkanRuntime::instance().active_subgroup_size();
}

XBT_EXPORT void xbt_vulkan_set_spirv_dir(const char* dir) {
    VulkanRuntime::instance().set_spirv_dir(dir);
}

XBT_EXPORT int32_t xbt_vulkan_load_spirv_module(
    int32_t kernel_id,
    const uint32_t* spirv_words,
    size_t byte_size
) {
    return VulkanRuntime::instance().load_spirv_bytes(kernel_id, spirv_words, byte_size);
}

XBT_EXPORT int32_t xbt_vulkan_eval_lj_dsf_csr(
    uint32_t n_atoms,
    uint32_t box_mode,
    uint32_t flags,
    const float* q_f4,
    const float* sig_eps,
    const uint32_t* sym_row_ptr,
    const uint32_t* sym_col_idx,
    uint32_t n_edges,
    const float* sym_wlj,
    const float* sym_wcoul,
    const float* dsf_table_f4,
    const float* box_ubo_32f,
    float rc2,
    float rcut,
    float inv_rc,
    float inv_rc2,
    float du,
    float inv_du,
    float coulomb_prefactor,
    float Arc,
    float Brc,
    float* out_forces_f4,
    float* out_u_half,
    float* out_vir9
) {
    return VulkanRuntime::instance().dispatch_lj_dsf_csr(
        n_atoms, box_mode, flags, q_f4, sig_eps, sym_row_ptr, sym_col_idx,
        n_edges, sym_wlj, sym_wcoul, dsf_table_f4, box_ubo_32f,
        rc2, rcut, inv_rc, inv_rc2, du, inv_du, coulomb_prefactor, Arc, Brc,
        out_forces_f4, out_u_half, out_vir9);
}

XBT_EXPORT int32_t xbt_vulkan_ternary_matvec(
    uint32_t M,
    uint32_t K_words,
    const uint64_t* W_P,
    const uint64_t* W_N,
    const uint64_t* x_P,
    const uint64_t* x_N,
    const float* row_scales,
    float x_scale,
    float* y_out
) {
    return VulkanRuntime::instance().dispatch_ternary_matvec(
        M, K_words, W_P, W_N, x_P, x_N, row_scales, x_scale, y_out);
}

XBT_EXPORT int32_t xbt_vulkan_msm_stencil27(
    uint32_t Ml,
    const float* q_in,
    const float* w27,
    uint32_t accumulate,
    float* phi_out
) {
    return VulkanRuntime::instance().dispatch_msm_stencil27(
        Ml, q_in, w27, accumulate, phi_out);
}
