#pragma once

#include <utils.hpp>

#include <random>
#include <utility>


// Constants definition
constexpr int D = 2 * 1; // Must be a multiple of 2

constexpr int MaskSize = 128; // Must be a power of two
constexpr int PixelCount = MaskSize * MaskSize;
constexpr int DistanceMatrixSize = PixelCount * (PixelCount + 1) / 2;

enum SamplerType {
   OWEN,
   RANK1,
};

class Optimizer {
public:
    /// \brief Default constructor.
    Optimizer(int spp, SamplerType type=SamplerType::RANK1);

    /// \brief Free the GL ressources before the destruction of the object.
    /// \note This is required because otherwise the context will be destroyed before the ressources are freed.
    void freeGLRessources();

    /// \brief Dispatch the compute shader.
    void run() const;

    /// \brief Starts the optimization of the next pair of dimensions.
    /// \return False if all the dimensions have already been optimized.
    bool nextDimensions();

    /// \brief Query the number of permutations that was accepted in all the compute shader dispatches.
    /// \return The number of permutations that was accepted in all the dispatches.
    uint32_t acceptedSwapCount() const;

    /// \brief Accessor for the display texture ID.
    /// \return The display texture OpenGL ID.
    GLuint displayTexture() const;

    /// \brief Export the latest mask as a header.
    /// \param filename The name of the file to export the mask in.
    void exportMaskAsHeader(const char *filename) const;

    /// \brief Export the latest mask as a ASCII tile.
    /// \param filename The name of the file to export the mask in.
    void exportMaskAsTile(const char *filename) const;

private:
    int m_dimension = 0;

    GLuint m_program;

    GLuint m_distanceMatrixSSBO;

    GLuint m_scramblesIn;

    GLuint m_scramblesOut;

    GLuint m_displayIn;

    GLuint m_displayOut;

    mutable std::mt19937 m_generator;

    GLuint m_permutationsSSBO;

    GLuint m_atomicCounter;

    std::vector<GLuint> m_scrambles;

    int m_spp;

    SamplerType m_type;

    //// Refactoring functions ////

    struct Heaviside {
        // Orientation vector
        float nx;
        float ny;

        // 2D point
        float px;
        float py;
    };

    /// \brief Generate the permutations that will be tested by the compute shader and store them in an SSBO.
    void generatePermutationsSSBO();

    /// \brief Generate the atomic coutner used to track the number of swaps in a single dispatch;
    void generateAtomicCounter();

    /// \brief Build the estimates matrix via calls to computeEstimatesMatrix and send it to the GPU.
    void setupTextures();

    /// \brief Generate an OpenGL RGBA32F 3D texture.
    /// \param internal_format The OpenGL internal format of the texture.
    /// \param format The OpenGL format of the texture.
    /// \param data_type The type of the data stored in the texture.
    /// \param image_unit The image unit to bind the texture to.
    /// \param access The type of access that will be done on the texture (e.g. GL_WRITE_ONLY).
    /// \param data The data to store in the texture shaped the way OpenGL expects it.
    /// \return The OpenGL texture ID.
    GLuint generateTexture(GLenum internal_format, GLenum format, GLenum data_type, int image_unit, GLenum access,
                           const void *data) const;

    /// \brief Generate the distance matrices and store them in an SSBO.
    /// \param scrambles The scrambling values for all the dimensions.
    void generateDistanceMatrix(GLuint *scrambles);

    /// \brief Preintegrate a given function (in that case, a 2D gaussian) that will be displayed.
    /// \param scrambling The scrambling values for all the dimensions.
    /// \return A vector containing the result.
    std::vector<GLfloat> preintegrateDisplay(GLuint *scrambling) const;

    /// \brief Integrate a 2D heaviside.
    /// \param scramble The scramble values to use for each dimensions.
    /// \param heaviside The 4 parameters that define an heaviside (i.e. an orientation vector and a 2D point).
    float integrateHeaviside(GLuint scramble[2], const Heaviside &heaviside) const;
};
