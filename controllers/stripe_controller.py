from datetime import datetime
from database import db
from database.models import PaymentTransaction, ThirdPartyAPICost
from fastapi import HTTPException
from typing import Optional
from uuid import uuid4 
import asyncio

def calculate_credits(amount):
    return amount * 1

async def get_user_by_customer_id(customer_id):
    user = await db.users_collection.find_one({"stripe_customer_id": customer_id})
    if user:
        return user
    else:
        raise HTTPException(status_code=404, detail="User not found")

async def add_credits_to_user(customer_id, credits):
    await db.users_collection.update_one(
                {"stripe_customer_id": customer_id},
                {"$set": {"total_credits": credits, "updated_at": datetime.utcnow()}}
            )
    return True

def is_upgrade(old_plan, new_plan):
    # Implement logic to determine if new_plan is an upgrade from old_plan
    # This could be based on plan IDs, prices, or a predefined hierarchy
    pass

async def handle_successful_subscription(session):

    customer_id = session['customer']
    amount_paid = session['amount_paid']
    subscription_id = session['subscription']
    
    # Fetch the user associated with this customer_id from your database
    user = get_user_by_customer_id(customer_id)
    
    if user:
        # Calculate credits based on amount paid or subscription plan
        credits_to_add = calculate_credits(amount_paid)
        
        # Add credits to the user's account
        add_credits_to_user(customer_id,credits_to_add)
        
        #log this transaction
        payment_transaction = PaymentTransaction(
                payment_id=str(uuid4()),
                user_id=user.get("user_id"),
                stripe_customer_id=user.get("stripe_customer_id"),
                payment_date= datetime.utcnow(),
                amount= amount_paid,
                status="success",
                subscription_id=user.get(subscription_id, None),
            )
        await db.payments_transaction_collection.insert_one(payment_transaction.dict())


async def handle_failed_payment(data):

    customer_id = data['customer']
    amount_paid = data['amount_paid']
    subscription_id = data['subscription']
    
    # Fetch the user associated with this customer_id from your database
    user = get_user_by_customer_id(customer_id)
    
    if user:
        
        # log this transaction
        payment_transaction = PaymentTransaction(
                payment_id=str(uuid4()),
                user_id=user.get("user_id"),
                stripe_customer_id=user.get("stripe_customer_id"),
                payment_date= datetime.utcnow(),
                amount= amount_paid,
                status="failed",
                subscription_id=user.get(subscription_id, None),
            )
        await db.payments_transaction_collection.insert_one(payment_transaction.dict())


async def handle_recurring_payment(invoice):
    # This handles successful recurring payments
    customer_id = invoice['customer']
    subscription_id = invoice['subscription']
    amount_paid = invoice['amount_paid']
    
    # Fetch the user associated with this customer_id from your database
    user = await get_user_by_customer_id(customer_id)
    
    if user:
         # Calculate credits based on amount paid or subscription plan
        credits_to_add = calculate_credits(amount_paid)

        # Add credits to the user's account
        add_credits_to_user(customer_id,credits_to_add)

        # log this transaction
        payment_transaction = PaymentTransaction(
                payment_id=str(uuid4()),
                user_id=user.get("user_id"),
                stripe_customer_id=user.get("stripe_customer_id"),
                payment_date= datetime.utcnow(),
                amount= amount_paid,
                status="success",
                subscription_id=user.get(subscription_id, None),
            )
        
        await db.payments_transaction_collection.insert_one(payment_transaction.dict())


async def handle_subscription_cancelled(subscription):
    # This handles when a subscription is cancelled
    customer_id = subscription['customer']
    
    # Fetch the user associated with this customer_id from your database
    user = await get_user_by_customer_id(customer_id)
    
    if user:
        # Update user's subscription status in your database
        await db.users_collection.update_one(
                {"stripe_customer_id": customer_id},
                {"$set": {"subscription_id": None, "updated_at": datetime.utcnow()}}
            )
        

async def handle_subscription_updated(subscription):
    customer_id = subscription['customer']
    user = await get_user_by_customer_id(customer_id)
    
    if not user:
        print(f"User not found for customer ID: {customer_id}")
        return

    old_plan =user["subscription_id"]
    new_plan = subscription['plan']['id']
    
    if old_plan and new_plan:
        if is_upgrade(old_plan, new_plan):
            await handle_subscription_upgrade(user, subscription)
        else:
            await handle_subscription_downgrade(user, subscription)
    else:
        print(f"Unable to determine if upgrade or downgrade for subscription: {subscription['id']}")


async def handle_subscription_upgrade(user, subscription):
    new_plan = subscription['plan']['id']
    additional_credits = calculate_additional_credits(new_plan)

    # Add credits immediately
    credits = user["remaining_credits"] + additional_credits
    await add_credits_to_user(user.id, credits)
    
    # Update user's subscription details
    await update_user_subscription(user.id, subscription['id'], 'active', new_plan)
    


async def handle_subscription_downgrade(user, subscription):
    new_plan = subscription['plan']['id']
    
    # Schedule credit adjustment for the next billing cycle
    next_billing_date = datetime.fromtimestamp(subscription['current_period_end'])
    await schedule_credit_adjustment(user.id, new_plan, next_billing_date)
    
    # Update user's subscription details
    await update_user_subscription(user.id, subscription['id'], 'active', new_plan)
