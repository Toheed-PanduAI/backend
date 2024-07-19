from datetime import datetime, timedelta
from database import db
from database.models import PaymentTransaction, ThirdPartyAPICost
from fastapi import HTTPException
from typing import Optional
from uuid import uuid4
import asyncio

PLAN_HIERARCHY = {"sample_free": 0, "sample_basic": 1, "sample_premium": 2}


def calculate_credits(amount):
    return amount * 1


def is_upgrade(old_plan, new_plan):
    # Check if both plans are valid
    if old_plan not in PLAN_HIERARCHY or new_plan not in PLAN_HIERARCHY:
        raise ValueError("Invalid plan")

    # Determine if the new plan is an upgrade from the old plan
    return PLAN_HIERARCHY[new_plan] > PLAN_HIERARCHY[old_plan]


async def schedule_downgrade(user_id, new_plan):
    await db.users_collection.update_one(
        {"user_id": user_id},
        {
            "$set": {
                "pending_downgrade": {
                    "new_plan": new_plan,
                    "apply_at": datetime.utcnow() + timedelta(days=30),
                }
            }
        },
    )


async def get_user_by_customer_id(customer_id):
    user = await db.users_collection.find_one({"stripe_customer_id": customer_id})
    if user:
        return user
    else:
        raise HTTPException(status_code=404, detail="User not found")


async def add_credits_to_user(customer_id, credits):
    await db.users_collection.update_one(
        {"stripe_customer_id": customer_id},
        {"$set": {"total_credits": credits, "updated_at": datetime.utcnow()}},
    )
    return True


async def update_user_subscription(user_id, subscription_id, plan_id):
    await db.users_collection.update_one(
        {"user_id": user_id},
        {
            "$set": {
                "plan_id": plan_id,
                "subscription_id": subscription_id,
                "updated_at": datetime.utcnow(),
            }
        },
    )
    return True


async def apply_pending_downgrades():
    current_time = datetime.utcnow()
    pending_downgrades = await db.users_collection.find(
        {"pending_downgrade.apply_at": {"$lte": current_time}}
    ).to_list(length=None)

    for user in pending_downgrades:
        new_plan = user["pending_downgrade"]["new_plan"]
        # Update user's plan
        await db.users_collection.update_one(
            {"user_id": user["user_id"]},
            {
                "$set": {"subscription_id": new_plan, "updated_at": datetime.utcnow()},
                "$unset": {"pending_downgrade": ""},
            },
        )
        print(f"Applied downgrade for user {user['user_id']} to plan {new_plan}.")


async def handle_successful_subscription(session):

    customer_id = session["customer"]
    amount_paid = session["amount_paid"]
    subscription_id = session["subscription"]
    plan_id = session["plan"]

    # Fetch the user associated with this customer_id from your database
    user = get_user_by_customer_id(customer_id)

    if user:
        # Update the user's subscription_id in your database
        await update_user_subscription(user["user_id"], subscription_id["id"], plan_id)

        # Calculate credits based on amount paid or subscription plan
        credits_to_add = calculate_credits(amount_paid)

        # Add credits to the user's account
        add_credits_to_user(customer_id, credits_to_add)

        # log this transaction
        payment_transaction = PaymentTransaction(
            payment_id=str(uuid4()),
            user_id=user.get("user_id"),
            stripe_customer_id=user.get("stripe_customer_id"),
            payment_date=datetime.utcnow(),
            amount=amount_paid,
            status="success",
            subscription_id=user.get(subscription_id, subscription_id),
        )
        await db.payments_transaction_collection.insert_one(payment_transaction.dict())


async def handle_failed_payment(data):

    customer_id = data["customer"]
    amount_paid = data["amount_paid"]
    subscription_id = data["subscription"]

    # Fetch the user associated with this customer_id from your database
    user = get_user_by_customer_id(customer_id)

    if user:

        # log this transaction
        payment_transaction = PaymentTransaction(
            payment_id=str(uuid4()),
            user_id=user.get("user_id"),
            stripe_customer_id=user.get("stripe_customer_id"),
            payment_date=datetime.utcnow(),
            amount=amount_paid,
            status="failed",
            subscription_id=user.get(subscription_id, None),
        )
        await db.payments_transaction_collection.insert_one(payment_transaction.dict())


async def handle_recurring_payment(invoice):
    # This handles successful recurring payments
    customer_id = invoice["customer"]
    subscription_id = invoice["subscription"]
    amount_paid = invoice["amount_paid"]

    # Fetch the user associated with this customer_id from your database
    user = await get_user_by_customer_id(customer_id)

    if user:
        # Calculate credits based on amount paid or subscription plan
        credits_to_add = calculate_credits(amount_paid)

        # Add credits to the user's account
        add_credits_to_user(customer_id, credits_to_add)

        # log this transaction
        payment_transaction = PaymentTransaction(
            payment_id=str(uuid4()),
            user_id=user.get("user_id"),
            stripe_customer_id=user.get("stripe_customer_id"),
            payment_date=datetime.utcnow(),
            amount=amount_paid,
            status="success",
            subscription_id=user.get(subscription_id, None),
        )

        await db.payments_transaction_collection.insert_one(payment_transaction.dict())


async def handle_subscription_cancelled(subscription):
    # This handles when a subscription is cancelled
    customer_id = subscription["customer"]

    # Fetch the user associated with this customer_id from your database
    user = await get_user_by_customer_id(customer_id)

    if user:
        # Update user's subscription status in your database
        await db.users_collection.update_one(
            {"stripe_customer_id": customer_id},
            {"$set": {"subscription_id": None, "updated_at": datetime.utcnow()}},
        )


async def handle_subscription_updated(subscription):
    customer_id = subscription["customer"]

    user = await get_user_by_customer_id(customer_id)

    if not user:
        print(f"User not found for customer ID: {customer_id}")
        return

    old_plan = user["subscription_id"]
    new_plan = subscription["plan"]["id"]

    if old_plan and new_plan:
        if is_upgrade(old_plan, new_plan):
            await handle_subscription_upgrade(user, customer_id, subscription)
        else:
            await handle_subscription_downgrade(user, customer_id, subscription)
    else:
        print(
            f"Unable to determine if upgrade or downgrade for subscription: {subscription['id']}"
        )


async def handle_subscription_upgrade(user, customer_id, subscription):
    plan_id = subscription["plan"]["id"]
    amount_paid = subscription["amount"]

    additional_credits = calculate_credits(amount_paid)

    # Add credits immediately
    credits = user["remaining_credits"] + additional_credits
    await add_credits_to_user(customer_id, credits)

    # Update user's subscription details
    await update_user_subscription(user["user_id"], subscription["id"], plan_id)
    # await update_user_subscription(user["user_id"], subscription['id'], 'active', new_plan)


async def handle_subscription_downgrade(user, subscription):
    plan_id = subscription["plan"]["id"]
    subscription_id = subscription["id"]

    await schedule_downgrade(user["user_id"], subscription_id)

    # Update user's subscription details
    await update_user_subscription(user["user_id"], subscription["id"], plan_id)
