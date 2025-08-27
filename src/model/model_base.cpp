#include "model/model_base.h"

#include "checkerboard/checkerboard_base.h"
#include "utils/assert.h"

namespace Model {

void ModelBase::mult_expK_from_left(GreensFunc& green) const {
  DQMC_ASSERT(green.rows() == this->m_space_size && green.cols() == this->m_space_size);
  m_temp_buffer.noalias() = this->m_expK_mat * green;
  green = m_temp_buffer;
}

void ModelBase::mult_expK_from_right(GreensFunc& green) const {
  DQMC_ASSERT(green.rows() == this->m_space_size && green.cols() == this->m_space_size);
  m_temp_buffer.noalias() = green * this->m_expK_mat;
  green = m_temp_buffer;
}

void ModelBase::mult_inv_expK_from_left(GreensFunc& green) const {
  DQMC_ASSERT(green.rows() == this->m_space_size && green.cols() == this->m_space_size);
  m_temp_buffer.noalias() = this->m_inv_expK_mat * green;
  green = m_temp_buffer;
}

void ModelBase::mult_inv_expK_from_right(GreensFunc& green) const {
  DQMC_ASSERT(green.rows() == this->m_space_size && green.cols() == this->m_space_size);
  m_temp_buffer.noalias() = green * this->m_inv_expK_mat;
  green = m_temp_buffer;
}

void ModelBase::mult_trans_expK_from_left(GreensFunc& green) const {
  DQMC_ASSERT(green.rows() == this->m_space_size && green.cols() == this->m_space_size);
  m_temp_buffer.noalias() = this->m_trans_expK_mat * green;
  green = m_temp_buffer;
}

void ModelBase::link() {
  this->m_mult_expK_from_left_ptr = &ModelBase::mult_expK_from_left;
  this->m_mult_expK_from_right_ptr = &ModelBase::mult_expK_from_right;
  this->m_mult_inv_expK_from_left_ptr = &ModelBase::mult_inv_expK_from_left;
  this->m_mult_inv_expK_from_right_ptr = &ModelBase::mult_inv_expK_from_right;
  this->m_mult_trans_expK_from_left_ptr = &ModelBase::mult_trans_expK_from_left;
  this->m_use_checkerboard = false;
  this->m_checkerboard_ptr = nullptr;
}

void ModelBase::link(const CheckerBoardBase& checkerboard) {
  this->m_mult_expK_from_left_ptr = &ModelBase::mult_expK_from_left_cb;
  this->m_mult_expK_from_right_ptr = &ModelBase::mult_expK_from_right_cb;
  this->m_mult_inv_expK_from_left_ptr = &ModelBase::mult_inv_expK_from_left_cb;
  this->m_mult_inv_expK_from_right_ptr = &ModelBase::mult_inv_expK_from_right_cb;
  this->m_mult_trans_expK_from_left_ptr = &ModelBase::mult_trans_expK_from_left_cb;
  this->m_use_checkerboard = true;
  this->m_checkerboard_ptr = &checkerboard;
}

void ModelBase::mult_expK_from_left_cb(GreensFunc& green) const {
  this->m_checkerboard_ptr->mult_expK_from_left(green);
}

void ModelBase::mult_expK_from_right_cb(GreensFunc& green) const {
  this->m_checkerboard_ptr->mult_expK_from_right(green);
}

void ModelBase::mult_inv_expK_from_left_cb(GreensFunc& green) const {
  this->m_checkerboard_ptr->mult_inv_expK_from_left(green);
}

void ModelBase::mult_inv_expK_from_right_cb(GreensFunc& green) const {
  this->m_checkerboard_ptr->mult_inv_expK_from_right(green);
}

void ModelBase::mult_trans_expK_from_left_cb(GreensFunc& green) const {
  this->m_checkerboard_ptr->mult_trans_expK_from_left(green);
}

}  // namespace Model
