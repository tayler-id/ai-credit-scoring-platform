"""Add MVP basic profile and declared income tables

Revision ID: 52c923d8f640
Revises: 458f710209f1
Create Date: 2025-04-28 12:44:29.790780

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '52c923d8f640'
down_revision: Union[str, None] = '458f710209f1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('mvp_basic_profiles',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('applicant_id', sa.String(), nullable=False),
    sa.Column('name', sa.String(), nullable=False),
    sa.Column('phone_number', sa.String(), nullable=False),
    sa.Column('occupation', sa.String(), nullable=True),
    sa.Column('years_in_business', sa.Integer(), nullable=True),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('applicant_id'),
    sa.UniqueConstraint('phone_number')
    )
    op.create_table('mvp_declared_income',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('applicant_id', sa.String(), nullable=False),
    sa.Column('monthly_income', sa.Float(), nullable=False),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('applicant_id')
    )
    # ### end Alembic commands ###


def downgrade() -> None:
    """Downgrade schema."""
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('mvp_declared_income')
    op.drop_table('mvp_basic_profiles')
    # ### end Alembic commands ###
